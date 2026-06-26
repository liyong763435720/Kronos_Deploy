import os

# 配置 Hugging Face 镜像站点（解决国内网络访问问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
import json
import threading
import plotly.graph_objects as go
import plotly.utils
import functools
import hashlib
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# ── Session 密钥（首次启动自动生成并持久化到配置文件）──────────

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Data source configuration (Tushare + TQSdk)
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasource_config.json')
_OLD_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tushare_config.json')
_SYMBOLS_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futures_symbols.json')

def _load_config() -> dict:
    """Load data source config, with backward-compat for old tushare_config.json."""
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    # migrate old token file
    if os.path.exists(_OLD_CONFIG_PATH):
        try:
            with open(_OLD_CONFIG_PATH, 'r', encoding='utf-8') as f:
                old = json.load(f)
                return {'tushare_token': old.get('token', '')}
        except Exception:
            pass
    return {}

def _save_config(cfg: dict):
    with open(_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False)

_cfg = _load_config()
TUSHARE_TOKEN: str = _cfg.get('tushare_token', '')
TQSDK_USERNAME: str = _cfg.get('tqsdk_username', '')
TQSDK_PASSWORD: str = _cfg.get('tqsdk_password', '')

# ── 访问密码（为空则不启用认证，本地使用时无需配置）────────────────
ACCESS_PASSWORD: str = _cfg.get('access_password', '')

# Session 密钥：首次启动自动生成并写入配置，重启后 session 仍有效
if not _cfg.get('secret_key'):
    _cfg['secret_key'] = os.urandom(32).hex()
    _save_config(_cfg)
app.secret_key = _cfg['secret_key']

# 服务端有效 token 集合，登出时从中移除，防止旧 cookie 重放
_valid_tokens: set = set()

def _hash_pwd(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def login_required(f):
    """路由装饰器：未配置密码时直接放行，配置了密码则验证 session + token。"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not ACCESS_PASSWORD:
            return f(*args, **kwargs)
        token = session.get('token')
        if token and token in _valid_tokens:
            return f(*args, **kwargs)
        # API 返回 401，页面跳转登录
        if request.path.startswith('/api/'):
            return jsonify({'error': '未登录', 'code': 401}), 401
        return redirect(url_for('login_page', next=request.path))
    return decorated

# 找到一个能正常 import tqsdk 的 Python 解释器（venv 里可能有安装问题）
def _find_tqsdk_python() -> str:
    import subprocess, shutil
    candidates = [
        shutil.which('python'),
        shutil.which('python3'),
        r'C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe',
        sys.executable,
    ]
    seen = set()
    for py in candidates:
        if not py or py in seen:
            continue
        seen.add(py)
        try:
            # TQSdk prints a disclaimer on first import, so allow 20 seconds
            r = subprocess.run([py, '-c', 'import tqsdk; print("ok")'],
                               capture_output=True, timeout=20)
            if r.returncode == 0 and b'ok' in r.stdout:
                return py
        except Exception:
            continue
    return sys.executable  # fallback

_TQSDK_PYTHON: str = _find_tqsdk_python()
print(f"[TQSdk] using Python: {_TQSDK_PYTHON}")

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def _to_tushare_code(stock_code: str) -> str:
    """Convert 6-digit A-share code to Tushare ts_code format (e.g. 600036 -> 600036.SH)."""
    code = stock_code.strip()
    if code.startswith('6') or code.startswith('9'):
        return f"{code}.SH"
    return f"{code}.SZ"

def ts_download_stock_daily(stock_code: str, days_back: int = 100):
    """Download A-share daily K-line via Tushare and return normalized DataFrame.

    Returns (DataFrame, None) on success, (None, error_message) on failure.
    """
    global TUSHARE_TOKEN
    if not TUSHARE_TOKEN:
        return None, "Tushare Token 未配置，请先在界面中设置 Token"

    try:
        import tushare as ts
        import pandas as pd
        import datetime as dt

        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=days_back * 2)

        ts.set_token(TUSHARE_TOKEN)
        ts_code = _to_tushare_code(stock_code)
        data = ts.pro_bar(
            ts_code=ts_code,
            adj='qfq',
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
        )

        if data is None or data.empty:
            return None, f"未找到股票 {stock_code} 的数据，请检查股票代码是否正确"

        # Tushare returns newest first, sort ascending
        data = data.sort_values('trade_date').reset_index(drop=True)

        data = data.rename(columns={
            'trade_date': 'Date',
            'vol': 'volume',
        })

        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

        # keep last N trading rows
        if len(data) > days_back:
            data = data.tail(days_back)

        data = data.copy()
        data['timestamps'] = data.index

        for c in ['open', 'high', 'low', 'close']:
            data[c] = pd.to_numeric(data[c], errors='coerce')
        if 'volume' in data.columns:
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
        if 'amount' in data.columns:
            data['amount'] = pd.to_numeric(data['amount'], errors='coerce')

        data = data.dropna(subset=['open', 'high', 'low', 'close'])
        if len(data) < min(30, days_back // 2):
            return None, f"清洗后数据不足: {len(data)} 行"

        return data, None
    except Exception as e:
        return None, f"Tushare 下载失败: {str(e)}"

# ── 期货数据（TQSdk） ────────────────────────────────────────────────────────

_FUTURES_EXCHANGES = {'SHFE', 'DCE', 'CZCE', 'CFFEX', 'INE', 'GFEX'}

# 品种代码 → 交易所映射（用于自动补全简短期货代码）
# SHFE/DCE/INE/GFEX 品种用小写；CZCE/CFFEX 品种用大写
_PRODUCT_EXCHANGE: dict = {
    # 上期所 SHFE（小写）
    'rb': 'SHFE', 'hc': 'SHFE', 'cu': 'SHFE', 'al': 'SHFE', 'zn': 'SHFE',
    'ni': 'SHFE', 'sn': 'SHFE', 'pb': 'SHFE', 'au': 'SHFE', 'ag': 'SHFE',
    'bu': 'SHFE', 'fu': 'SHFE', 'sp': 'SHFE', 'ss': 'SHFE', 'wr': 'SHFE',
    'ru': 'SHFE', 'ao': 'SHFE', 'ec': 'SHFE',
    # 大商所 DCE（小写）
    'm': 'DCE', 'a': 'DCE', 'b': 'DCE', 'c': 'DCE', 'cs': 'DCE',
    'jd': 'DCE', 'l': 'DCE', 'p': 'DCE', 'v': 'DCE', 'y': 'DCE',
    'i': 'DCE', 'j': 'DCE', 'jm': 'DCE', 'eg': 'DCE', 'rr': 'DCE',
    'pg': 'DCE', 'eb': 'DCE', 'lh': 'DCE', 'bb': 'DCE', 'fb': 'DCE', 'pp': 'DCE', 'bz': 'DCE',
    # 郑商所 CZCE（大写，3位交割月）
    'SR': 'CZCE', 'CF': 'CZCE', 'RM': 'CZCE', 'MA': 'CZCE', 'OI': 'CZCE',
    'TA': 'CZCE', 'ZC': 'CZCE', 'FG': 'CZCE', 'AP': 'CZCE', 'CJ': 'CZCE',
    'PK': 'CZCE', 'SM': 'CZCE', 'SF': 'CZCE', 'UR': 'CZCE', 'SA': 'CZCE',
    'PF': 'CZCE', 'RS': 'CZCE', 'WH': 'CZCE', 'PM': 'CZCE', 'RI': 'CZCE',
    'LR': 'CZCE', 'JR': 'CZCE', 'CY': 'CZCE', 'SH': 'CZCE', 'PX': 'CZCE',
    # 中金所 CFFEX（大写）
    'IF': 'CFFEX', 'IC': 'CFFEX', 'IH': 'CFFEX', 'IM': 'CFFEX',
    'TF': 'CFFEX', 'T': 'CFFEX', 'TS': 'CFFEX', 'TL': 'CFFEX',
    # 上期能源 INE（小写）
    'sc': 'INE', 'lu': 'INE', 'nr': 'INE', 'bc': 'INE',
    # 广期所 GFEX（小写）
    'si': 'GFEX', 'lc': 'GFEX', 'ps': 'GFEX', 'br': 'GFEX',
}

def _normalize_futures_code(code: str) -> str:
    """将简短期货代码扩展为 TQSdk 完整格式。
    rb2610  → SHFE.rb2610
    SR607   → CZCE.SR607
    已有前缀（SHFE.rb2610 / KQ.m@SHFE.rb）则直接返回。
    """
    import re
    code = code.strip()
    if '.' in code:
        return code
    m = re.match(r'^([A-Za-z]+)(\d+)$', code)
    if not m:
        return code
    product, month = m.group(1), m.group(2)
    # 查映射：先原始大小写，再大写，再小写
    exchange = (_PRODUCT_EXCHANGE.get(product)
                or _PRODUCT_EXCHANGE.get(product.upper())
                or _PRODUCT_EXCHANGE.get(product.lower()))
    if not exchange:
        return code
    # 规范化大小写：CZCE/CFFEX 大写，其余小写
    if exchange in ('CZCE', 'CFFEX'):
        product = product.upper()
    else:
        product = product.lower()
    return f"{exchange}.{product}{month}"

# 品种代码 → 中文名称
_PRODUCT_NAME: dict = {
    # 上期所 SHFE
    'rb': '螺纹钢', 'hc': '热轧卷板', 'cu': '铜', 'al': '铝', 'zn': '锌',
    'ni': '镍', 'sn': '锡', 'pb': '铅', 'au': '黄金', 'ag': '白银',
    'bu': '沥青', 'fu': '燃料油', 'sp': '纸浆', 'ss': '不锈钢', 'wr': '线材',
    'ru': '天然橡胶', 'ao': '氧化铝', 'ec': '集运欧线',
    # 大商所 DCE
    'm': '豆粕', 'a': '豆一', 'b': '豆二', 'c': '玉米', 'cs': '玉米淀粉',
    'jd': '鸡蛋', 'l': '塑料', 'p': '棕榈油', 'v': 'PVC', 'y': '豆油',
    'i': '铁矿石', 'j': '焦炭', 'jm': '焦煤', 'eg': '乙二醇', 'rr': '粳稻',
    'pg': '液化气', 'eb': '苯乙烯', 'lh': '生猪', 'bb': '胶合板', 'fb': '纤维板', 'pp': '聚丙烯', 'bz': '纯苯',
    # 郑商所 CZCE（大写）
    'SR': '白糖', 'CF': '棉花', 'RM': '菜粕', 'MA': '甲醇', 'OI': '菜油',
    'TA': 'PTA', 'ZC': '动力煤', 'FG': '玻璃', 'AP': '苹果', 'CJ': '红枣',
    'PK': '花生', 'SM': '锰硅', 'SF': '硅铁', 'UR': '尿素', 'SA': '纯碱',
    'PF': '涤纶短纤', 'RS': '菜籽', 'WH': '强筋小麦', 'PM': '普通小麦',
    'RI': '早稻', 'LR': '晚稻', 'JR': '粳稻', 'CY': '棉纱', 'SH': '烧碱', 'PX': '对二甲苯',
    # 中金所 CFFEX（大写）
    'IF': '沪深300指数', 'IC': '中证500指数', 'IH': '上证50指数', 'IM': '中证1000指数',
    'TF': '5年国债', 'T': '10年国债', 'TS': '2年国债', 'TL': '30年国债',
    # 上期能源 INE
    'sc': '原油', 'lu': '低硫燃料油', 'nr': '20号胶', 'bc': '国际铜',
    # 广期所 GFEX
    'si': '工业硅', 'lc': '碳酸锂', 'ps': '多晶硅', 'br': '丁二烯橡胶',
}

def _load_symbols_cache():
    """启动时加载 futures_symbols.json，将缓存内容合并进全局映射表。
    缓存优先级高于硬编码，新品种上市后刷新一次即可永久生效。
    """
    if not os.path.exists(_SYMBOLS_CACHE_PATH):
        return
    try:
        with open(_SYMBOLS_CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        exchange_map = cache.get('exchange', {})
        name_map = cache.get('name', {})
        _PRODUCT_EXCHANGE.update(exchange_map)
        _PRODUCT_NAME.update(name_map)
        print(f"[symbols] 已从缓存加载 {len(exchange_map)} 个品种")
    except Exception as e:
        print(f"[symbols] 加载缓存失败: {e}")

# 启动时合并缓存
_load_symbols_cache()


def _refresh_symbols_from_tqsdk():
    """通过 TQSdk 子进程拉取全市场期货品种列表，写入本地缓存并更新运行时映射表。
    Returns: (count: int, error: str|None)
    """
    global TQSDK_USERNAME, TQSDK_PASSWORD
    if not TQSDK_USERNAME or not TQSDK_PASSWORD:
        return 0, "TQSdk 账号未配置，请先在数据源设置中填写账号"

    import subprocess, tempfile, re as _re

    script = f"""
import sys, warnings, asyncio, time, json, re
warnings.filterwarnings('ignore')

username = {repr(TQSDK_USERNAME)}
password = {repr(TQSDK_PASSWORD)}

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    from tqsdk import TqApi, TqAuth
    api = TqApi(auth=TqAuth(username, password))

    # TQSdk 3.x: query_symbol_info 不再接受 ins_class 参数
    # 改用 query_quotes 获取合约列表，再从代码中解析品种信息
    symbols_task = api.query_quotes(ins_class="FUTURE", expired=False)
    deadline = time.time() + 60
    while not symbols_task._task.done():
        if not api.wait_update(deadline=deadline):
            api.close()
            sys.stderr.write("TIMEOUT")
            sys.exit(1)

    all_symbols = list(symbols_task)
    api.close()

    # 从合约代码 EXCHANGE.PRODUCTyyyymm 中解析品种和交易所
    # 例: SHFE.rb2507 -> exchange=SHFE, product=rb
    result = {{}}
    for sym in all_symbols:
        parts = sym.split('.')
        if len(parts) != 2:
            continue
        exchange_id, inst = parts
        product_id = re.sub(r'\\d+$', '', inst).strip()
        if product_id and product_id not in result:
            result[product_id] = {{'exchange': exchange_id, 'name': ''}}

    print(json.dumps(result, ensure_ascii=False))
except Exception as e:
    import traceback
    sys.stderr.write(traceback.format_exc())
    sys.exit(1)
finally:
    try:
        loop.close()
    except Exception:
        pass
"""

    tmp = tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8')
    tmp.write(script)
    tmp.close()

    try:
        proc = subprocess.run(
            [_TQSDK_PYTHON, tmp.name],
            capture_output=True, text=True, timeout=90
        )
        if proc.returncode != 0:
            # 取第一个 Error/Exception 行，避免被 websocket 清理噪音干扰
            lines = (proc.stderr or 'unknown').strip().splitlines()
            err = next((l for l in lines if 'Error' in l or 'Exception' in l), lines[-1] if lines else 'unknown')
            return 0, f"TQSdk 查询失败: {err}"

        # TQSdk 日志可能混入 stdout，找最后一行以 { 开头的 JSON
        json_line = next((l for l in reversed(proc.stdout.splitlines()) if l.strip().startswith('{')), None)
        if not json_line:
            return 0, "未获取到品种数据"
        data = json.loads(json_line)
        if not data:
            return 0, "未获取到品种数据"

        exchange_map = {p: v['exchange'] for p, v in data.items()}
        name_map     = {p: v['name']     for p, v in data.items() if v.get('name')}

        cache = {
            'exchange':    exchange_map,
            'name':        name_map,
            'updated_at':  datetime.datetime.now().isoformat(),
        }
        with open(_SYMBOLS_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        _PRODUCT_EXCHANGE.update(exchange_map)
        _PRODUCT_NAME.update(name_map)
        return len(data), None

    except json.JSONDecodeError as e:
        return 0, f"解析响应失败: {e}"
    except subprocess.TimeoutExpired:
        return 0, "TQSdk 查询超时（90s）"
    except Exception as e:
        return 0, f"刷新异常: {str(e)}"
    finally:
        os.unlink(tmp.name)

def _futures_display_name(original_code: str) -> str:
    """返回期货合约的友好显示名称，用于图表标题。
    RS609       → 'RS609 菜籽'
    rb2610      → 'rb2610 螺纹钢'
    SHFE.rb2610 → 'rb2610 螺纹钢'
    KQ.m@SHFE.rb→ 'KQ.m@SHFE.rb 豆粕主力'
    """
    import re
    code = original_code.strip()
    full = _normalize_futures_code(code)

    # 主力/指数连续合约：KQ.m@SHFE.rb
    if full.upper().startswith('KQ.'):
        m = re.search(r'@\w+\.([A-Za-z]+)$', full)
        if m:
            p = m.group(1)
            name = (_PRODUCT_NAME.get(p)
                    or _PRODUCT_NAME.get(p.upper())
                    or _PRODUCT_NAME.get(p.lower()) or '')
            suffix = '主力' if '.m@' in full.lower() else '指数'
            return f"{code} {name}{suffix}" if name else code
        return code

    # 有交易所前缀：SHFE.rb2610 → 取 rb2610 部分
    contract = full.split('.', 1)[1] if '.' in full else full
    m = re.match(r'^([A-Za-z]+)(\d+)$', contract)
    if not m:
        return code
    product, month_digits = m.group(1), m.group(2)
    name = (_PRODUCT_NAME.get(product)
            or _PRODUCT_NAME.get(product.upper())
            or _PRODUCT_NAME.get(product.lower()) or '')
    # 显示时用用户原始输入的代码（简短或完整均可）+中文名
    display_code = contract  # 去掉交易所前缀，更简洁
    return f"{display_code} {name}" if name else display_code

def is_futures_code(code: str) -> bool:
    """Return True if code looks like a TQSdk futures symbol.

    Supported formats:
      SHFE.rb2507          — specific contract with exchange prefix
      KQ.m@SHFE.rb         — main continuous contract
      KQ.i@SHFE.rb         — index continuous contract
      rb2610               — short form (auto-expands to SHFE.rb2610)
      SR607 / PK610        — CZCE short form
    """
    import re
    upper = code.upper()
    # KQ.m@ / KQ.i@ continuous contract
    if upper.startswith('KQ.'):
        return True
    # exchange.product format
    parts = upper.split('.')
    if len(parts) == 2 and parts[0] in _FUTURES_EXCHANGES:
        return True
    # Short form: letters + digits (no dot), product must be in mapping
    m = re.match(r'^([A-Za-z]+)\d+$', code)
    if m:
        p = m.group(1)
        return bool(_PRODUCT_EXCHANGE.get(p)
                    or _PRODUCT_EXCHANGE.get(p.upper())
                    or _PRODUCT_EXCHANGE.get(p.lower()))
    return False

# K线周期秒数映射
KLINE_PERIODS = {'1d': 86400, '1h': 3600, '30m': 1800, '15m': 900}


def _resample_to_trading_bars(df_1min: pd.DataFrame, period_minutes: int) -> pd.DataFrame:
    """将1分钟K线按交易时间重采样为N分钟K线。

    判断规则：
    - gap < period_minutes 分钟 → 小休（如大商所10:15-10:30），继续当前bar计数
    - gap >= period_minutes 分钟 → 新session，重置bar计数（午休/收盘/隔夜均触发）

    这样 10:00-10:14（15根）+ 10:30-10:44（15根）自动合并为一根30分K，
    时间戳为最后一根1分K的时间+1分钟（=K线收盘时间），与行情软件一致。
    """
    df = df_1min.sort_index().copy()

    # gap >= 4小时才算新session（小歇15min和午休2h均不打断bar计数，
    # 只有收盘→夜盘（≥6h）或隔夜（≥6.5h）才重置）
    times = df.index.to_series()
    is_new_session = times.diff() >= pd.Timedelta(hours=4)
    session_id = is_new_session.cumsum()

    # session内累计编号，每 period_minutes 根为一组
    within_idx = df.groupby(session_id).cumcount()
    bar_num = within_idx // period_minutes
    group_key = session_id * 100000 + bar_num

    # 聚合 OHLCV
    agg = df.groupby(group_key).agg(
        open=('open',  'first'),
        high=('high',  'max'),
        low= ('low',   'min'),
        close=('close','last'),
    )
    if 'volume' in df.columns:
        agg['volume'] = df.groupby(group_key)['volume'].sum()

    # 时间戳 = 每组最后一根1分K的起始时间 + 1分钟（= K线收盘时间）
    # TQSdk 1分K时间戳是bar起始时间，+1分钟才是该分钟的收盘
    bar_end = df.groupby(group_key).apply(lambda x: x.index[-1]) + pd.Timedelta(minutes=1)
    agg.index = bar_end.values
    agg.index.name = 'Date'

    return agg.sort_index()


def tq_download_futures_daily(symbol: str, days_back: int = 200, period: str = '1d'):
    """Download futures K-line via TQSdk using a subprocess to avoid
    asyncio/Flask event-loop conflicts.

    period: '1d'=日K, '1h'=1小时K, '30m'=30分钟K, '15m'=15分钟K
    days_back: 日K时为天数，分钟K时为K线条数
    Returns (DataFrame, None) on success, (None, error_message) on failure.
    """
    global TQSDK_USERNAME, TQSDK_PASSWORD
    if not TQSDK_USERNAME or not TQSDK_PASSWORD:
        return None, "TQSdk 账号未配置，请先在界面中设置用户名和密码"

    import subprocess, tempfile, sys, os

    period_seconds = KLINE_PERIODS.get(period, 86400)

    if period_seconds < 86400:
        # 分钟K：下载1分钟原始数据，本地重采样，确保与行情软件K线边界一致
        period_minutes = period_seconds // 60
        dl_period_seconds = 60
        dl_count = min(days_back * period_minutes + 500, 15000)
    else:
        # 日K：直接下载目标周期
        period_minutes = None
        dl_period_seconds = period_seconds
        dl_count = min(days_back * 2, 8000)

    # 独立 Python 脚本，在子进程中跑 TQSdk，结果写入临时 CSV
    script = f"""
import sys, warnings
warnings.filterwarnings('ignore')
import asyncio, time, pandas as pd

username = {repr(TQSDK_USERNAME)}
password = {repr(TQSDK_PASSWORD)}
symbol   = {repr(symbol)}
count    = {dl_count}
out_csv  = sys.argv[1]

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    from tqsdk import TqApi, TqAuth
    api = TqApi(auth=TqAuth(username, password))
    klines = api.get_kline_serial(symbol, {dl_period_seconds}, data_length=count)
    deadline = time.time() + 60
    while not api.is_serial_ready(klines):
        if not api.wait_update(deadline=deadline):
            api.close()
            sys.stderr.write(f"TIMEOUT: {{symbol}}")
            sys.exit(1)
    pd.DataFrame(klines).to_csv(out_csv, index=False)
    api.close()
except Exception as e:
    import traceback
    sys.stderr.write(traceback.format_exc())
    sys.exit(1)
finally:
    loop.close()
"""

    tmp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_csv.close()
    tmp_py  = tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8')
    tmp_py.write(script)
    tmp_py.close()

    try:
        proc = subprocess.run(
            [_TQSDK_PYTHON, tmp_py.name, tmp_csv.name],
            capture_output=True, text=True, timeout=90
        )
        if proc.returncode != 0:
            err = (proc.stderr or 'unknown error').strip().splitlines()[-1]
            return None, f"TQSdk 下载失败: {err}"

        df = pd.read_csv(tmp_csv.name)
    except subprocess.TimeoutExpired:
        return None, "TQSdk 下载超时（90s）"
    except Exception as e:
        return None, f"TQSdk 子进程异常: {str(e)}"
    finally:
        os.unlink(tmp_py.name)
        try:
            os.unlink(tmp_csv.name)
        except Exception:
            pass
    # datetime field is int64 nanoseconds UTC（TQSdk 存的是 bar 起始时间）
    df['Date'] = (pd.to_datetime(df['datetime'], unit='ns', utc=True)
                  .dt.tz_convert('Asia/Shanghai')
                  .dt.tz_localize(None))
    df = df.sort_values('Date').set_index('Date')

    # drop rows where all OHLC are 0 (unfinished/empty bars)
    df = df[(df['open'] != 0) | (df['close'] != 0)]

    if period_minutes is not None:
        # 按交易时间重采样：自动处理休市gap，K线边界与行情软件一致
        df = _resample_to_trading_bars(df, period_minutes)

    if len(df) > days_back:
        df = df.tail(days_back)

    df = df.copy()
    df['timestamps'] = df.index

    for c in ['open', 'high', 'low', 'close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    if len(df) < min(30, days_back // 2):
        return None, f"清洗后数据不足: {len(df)} 行"

    return df, None

def download_market_data(code: str, days_back: int = 200, period: str = '1d'):
    """Auto-route to stock (Tushare) or futures (TQSdk) downloader by code format."""
    if is_futures_code(code):
        full_code = _normalize_futures_code(code)
        return tq_download_futures_daily(full_code, days_back=days_back, period=period)
    return ts_download_stock_daily(code, days_back=days_back)

def simple_prediction(data, pred_trading_days=22):
    """Simple technical analysis prediction as fallback when Kronos model is not available."""
    import numpy as np
    import pandas as pd
    
    # Calculate technical indicators
    df = data.copy()
    
    # Moving averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Get latest values
    latest = df.iloc[-1]
    current_price = latest['close']
    
    # Trend analysis
    if latest['MA5'] > latest['MA10'] > latest['MA20']:
        trend_factor = 1.02  # Upward trend
    elif latest['MA5'] < latest['MA10'] < latest['MA20']:
        trend_factor = 0.98  # Downward trend
    else:
        trend_factor = 1.0   # Sideways
    
    # RSI analysis
    if latest['RSI'] > 70:
        rsi_factor = 0.99  # Overbought
    elif latest['RSI'] < 30:
        rsi_factor = 1.01  # Oversold
    else:
        rsi_factor = 1.0   # Normal
    
    # Generate predictions
    predictions = []
    for i in range(pred_trading_days):
        # Add random noise
        random_factor = np.random.normal(1.0, 0.02)
        
        # Calculate predicted price (compound step-by-step, not exponential of i)
        predicted_price = current_price * trend_factor * rsi_factor * random_factor
        
        # Generate OHLC
        high = predicted_price * (1 + abs(np.random.normal(0, 0.01)))
        low = predicted_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = predicted_price * (1 + np.random.normal(0, 0.005))
        volume = latest.get('volume', 1000000) * (1 + np.random.normal(0, 0.1))
        
        predictions.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': predicted_price,
            'volume': volume,
            'amount': volume * predicted_price
        })
        
        current_price = predicted_price
    
    # Create DataFrame with proper index
    pred_df = pd.DataFrame(predictions)
    pred_df.index = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=pred_trading_days)
    
    return pred_df

def load_data_files():
    """Scan data directory and return available data files"""
    # Check multiple possible data directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_dirs = [
        os.path.join(base_dir, 'data'),
        os.path.join(base_dir, 'examples', 'data'),
        os.path.join(base_dir, 'finetune_csv', 'data')
    ]
    
    data_files = []
    
    for data_dir in possible_dirs:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(('.csv', '.feather')):
                    file_path = os.path.join(data_dir, file)
                    file_size = os.path.getsize(file_path)
                    data_files.append({
                        'name': file,
                        'path': file_path,
                        'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                    })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows where required OHLC columns have NaN
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        # Remove duplicate timestamps (keep last entry, e.g. intraday partial bar)
        if 'timestamps' in df.columns:
            df = df.drop_duplicates(subset=['timestamps'], keep='last').reset_index(drop=True)

        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
                first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

@app.route('/api/predict-by-code', methods=['POST'])
@login_required
def predict_by_code():
    """Predict by A-share stock code: downloads last N trading days (user-configurable, default 100) and predicts next ~1 month (approx. 22 bdays)."""
    global predictor
    
    try:
        data = request.get_json() or {}
        stock_code = data.get('stock_code', '').strip()
        lookback = int(data.get('lookback', 100))
        pred_count = int(data.get('pred_trading_days', 22))
        kline_period = data.get('kline_period', '1d')  # '1d','1h','30m','15m'

        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        if not stock_code:
            return jsonify({'error': '代码不能为空'}), 400

        # Detect instrument type and normalize futures code
        is_futures = is_futures_code(stock_code)
        display_code = _normalize_futures_code(stock_code) if is_futures else stock_code
        # instrument_label 用于图表标题：期货显示"rb2610 螺纹钢"，股票显示代码
        if is_futures:
            instrument_label = _futures_display_name(stock_code)
        else:
            instrument_label = stock_code

        # Check if Kronos model is available, if not use simple prediction
        use_kronos = MODEL_AVAILABLE and predictor is not None

        # 分钟K仅期货支持
        if kline_period != '1d' and not is_futures_code(stock_code):
            return jsonify({'error': '分钟K线仅支持期货代码，股票请使用日K'}), 400

        # 1) Download data (auto-routes: futures → TQSdk, stock → Tushare)
        df, err = download_market_data(stock_code, days_back=lookback, period=kline_period)
        if err:
            return jsonify({'error': f'下载数据失败: {err}'}), 400

        # 2) Prepare x/y timestamps
        import pandas as pd
        from pandas.tseries.offsets import BDay

        required_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            required_cols.append('volume')
        if 'amount' in df.columns:
            required_cols.append('amount')
        x_df = df[required_cols].copy()
        x_timestamp = pd.Series(df['timestamps'].values, name='timestamps')

        # 自适应生成未来时间戳：日K用交易日，分钟K从历史数据推算间隔
        last_date = df.index[-1]
        if kline_period == '1d':
            future_ts = pd.bdate_range(start=last_date + BDay(1), periods=pred_count)
        else:
            recent_idx = df.index[-min(30, len(df)):]
            median_interval = pd.Series(recent_idx).diff().dropna().median()
            future_ts = pd.date_range(start=last_date + median_interval,
                                      periods=pred_count, freq=median_interval)
        y_timestamp = pd.Series(future_ts, name='timestamps')

        # 3) Call predictor (Kronos or simple)
        if use_kronos:
            try:
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=len(y_timestamp),
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=False,
                )
            except Exception as e:
                return jsonify({'error': f'Kronos模型预测失败: {str(e)}'}), 500
        else:
            # Use simple technical analysis prediction
            try:
                pred_df = simple_prediction(x_df, pred_count)
            except Exception as e:
                return jsonify({'error': f'简化预测失败: {str(e)}'}), 500

        # 4) Build chart JSON with historical and prediction
        chart_json = create_prediction_chart(
            df.reset_index(),  # expects timestamps column or index
            pred_df,
            lookback=len(x_df),
            pred_len=len(y_timestamp),
            actual_df=None,
            historical_start_idx=max(0, len(df) - len(x_df)),
            label=instrument_label,
        )

        # 5) Build response prediction list
        prediction_results = []
        for i, (ts, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0.0)),
                'amount': float(row.get('amount', 0.0)),
            })

        # 6) Save json snapshot
        try:
            save_prediction_results(
                file_path=f'stock_code:{stock_code}',
                prediction_type=f'Kronos按{instrument_label}代码预测（{lookback}条历史 + {pred_count}条预测，周期{kline_period}）',
                prediction_results=prediction_results,
                actual_data=[],
                input_data=x_df,
                prediction_params={
                    'stock_code': stock_code,
                    'lookback': lookback,
                    'pred_trading_days': pred_count,
                    'kline_period': kline_period,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                },
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")

        prediction_mode = "Kronos AI模型" if use_kronos else "简化技术分析"
        return jsonify({
            'success': True,
            'stock_code': display_code,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'message': f'[{instrument_label}] {display_code} — 已使用{prediction_mode}预测，{kline_period} 周期，基于最近 {lookback} 条历史数据生成未来 {pred_count} 条预测结果'
        })
    except Exception as e:
        return jsonify({'error': f'服务异常: {str(e)}'}), 500

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0, label='股票'):
    """Create prediction chart with subplots for price and volume"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
    
    # Create subplots with 3 rows: price, volume, and empty space
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{label}价格预测', '成交量预测', f'{label}K线图'),
        vertical_spacing=0.07,
        row_heights=[0.45, 0.25, 0.30]
    )
    
    # Add historical price data (line chart for better visibility)
    hist_timestamps = historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index
    fig.add_trace(go.Scatter(
        x=hist_timestamps,
        y=historical_df['close'],
        mode='lines',
        name=f'{label} 历史收盘价',
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)

    # Add candlestick to third subplot
    fig.add_trace(go.Candlestick(
        x=hist_timestamps,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name=f'{label} 历史K线',
        increasing_line_color='#FF4D4F',
        increasing_fillcolor='#FF4D4F',
        decreasing_line_color='#52C41A',
        decreasing_fillcolor='#52C41A',
        whiskerwidth=0.2,
        opacity=0.9
    ), row=3, col=1)
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].dropna().diff().median() if len(df) > 1 else pd.Timedelta(hours=1)

            # For daily data use business days to avoid weekend gaps; for intraday use fixed freq
            if time_diff >= pd.Timedelta(days=1):
                from pandas.tseries.offsets import BDay
                pred_timestamps = pd.bdate_range(start=last_timestamp + BDay(1), periods=len(pred_df))
            else:
                pred_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=len(pred_df),
                    freq=time_diff
                )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        # Respect user's preference: only show predicted volume if provided by the model

        # Add prediction price data (line chart)
        fig.add_trace(go.Scatter(
            x=pred_timestamps,
            y=pred_df['close'],
            mode='lines',
            name=f'{label} 预测收盘价',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ), row=1, col=1)

        # Add predicted candlestick to third subplot (if OHLC provided)
        if {'open','high','low','close'}.issubset(set(pred_df.columns)):
            fig.add_trace(go.Candlestick(
                x=pred_timestamps,
                open=pred_df['open'],
                high=pred_df['high'],
                low=pred_df['low'],
                close=pred_df['close'],
                name=f'{label} 预测K线',
                increasing_line_color='#FF4D4F',
                increasing_fillcolor='rgba(255,77,79,0.4)',
                decreasing_line_color='#52C41A',
                decreasing_fillcolor='rgba(82,196,26,0.4)',
                whiskerwidth=0.2,
                opacity=0.7
            ), row=3, col=1)
        
        # Add historical volume data (bar chart)
        if 'volume' in historical_df.columns:
            fig.add_trace(go.Bar(
                x=hist_timestamps,
                y=historical_df['volume'],
                name='历史成交量',
                marker_color='#1f77b4',
                opacity=0.7
            ), row=2, col=1)
        
        # Add prediction volume data (bar chart)
        if 'volume' in pred_df.columns:
            fig.add_trace(go.Bar(
                x=pred_timestamps,
                y=pred_df['volume'],
                name='预测成交量',
                marker_color='#ff7f0e',
                opacity=0.7
            ), row=2, col=1)
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        # Add actual price data
        fig.add_trace(go.Scatter(
            x=actual_timestamps,
            y=actual_df['close'],
            mode='lines',
            name='实际收盘价',
            line=dict(color='#2ca02c', width=2)
        ), row=1, col=1)
        
        # Add actual volume data
        if 'volume' in actual_df.columns:
            fig.add_trace(go.Bar(
                x=actual_timestamps,
                y=actual_df['volume'],
                name='实际成交量',
                marker_color='#2ca02c',
                opacity=0.7
            ), row=2, col=1)

        # Add actual candlestick to third subplot
        if {'open','high','low','close'}.issubset(set(actual_df.columns)):
            fig.add_trace(go.Candlestick(
                x=actual_timestamps,
                open=actual_df['open'],
                high=actual_df['high'],
                low=actual_df['low'],
                close=actual_df['close'],
                name='实际K线',
                increasing_line_color='#FAAD14',
                increasing_fillcolor='rgba(250,173,20,0.5)',
                decreasing_line_color='#FAAD14',
                decreasing_fillcolor='rgba(250,173,20,0.5)',
                whiskerwidth=0.2,
                opacity=0.6
            ), row=3, col=1)
    
    # Update layout for subplots
    fig.update_layout(
        title='价格/成交量/日K 预测结果',
        template='plotly_dark',
        paper_bgcolor='#0b1220',
        plot_bgcolor='#0b1220',
        height=1000,
        width=1400,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=12, color='#e5e7eb'),
        title_font_size=16,
    )
    
    # 所有子图 x 轴使用 category 类型，消除周末/节假日空白
    fig.update_xaxes(
        type='category',
        tickformat="%Y-%m-%d",
        rangeslider_visible=False,
        nticks=12,
    )
    fig.update_yaxes(title_text="价格 (元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量",   row=2, col=1)
    fig.update_yaxes(title_text="价格 (元)", row=3, col=1)
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ── 登录 / 登出 / 改密码 ───────────────────────────────────────────

@app.route('/login')
def login_page():
    token = session.get('token')
    if not ACCESS_PASSWORD or (token and token in _valid_tokens):
        return redirect('/')
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    global ACCESS_PASSWORD
    body = request.get_json() or {}
    pwd = body.get('password', '')
    if ACCESS_PASSWORD and _hash_pwd(pwd) == _hash_pwd(ACCESS_PASSWORD):
        token = os.urandom(16).hex()
        _valid_tokens.add(token)
        session['token'] = token
        session.permanent = False
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '密码错误'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    token = session.get('token')
    _valid_tokens.discard(token)
    session.clear()
    return jsonify({'success': True})

@app.route('/api/access-password', methods=['GET'])
@login_required
def get_access_password():
    return jsonify({'enabled': bool(ACCESS_PASSWORD)})

@app.route('/api/access-password', methods=['POST'])
@login_required
def set_access_password():
    global ACCESS_PASSWORD
    body = request.get_json() or {}
    new_pwd = body.get('password', '').strip()
    cfg = _load_config()
    cfg['access_password'] = new_pwd
    _save_config(cfg)
    ACCESS_PASSWORD = new_pwd
    return jsonify({'success': True, 'message': '密码已更新' if new_pwd else '访问密码已关闭'})

# ─────────────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
@login_required
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
@login_required
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))

        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        # Convert start_date once here, reuse throughout
        start_date_raw = data.get('start_date')
        start_dt = pd.to_datetime(start_date_raw) if start_date_raw else None
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection (backtesting mode)
                if start_dt is not None:
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'从 {start_dt.strftime("%Y-%m-%d")} 开始数据不足，需要 {lookback + pred_len} 条，当前只有 {len(time_range_df)} 条'}), 400
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = f"回测验证：从 {start_dt.strftime('%Y-%m-%d')} 起前 {lookback} 条数据预测，后 {pred_len} 条真实数据对比"
                else:
                    if len(df) < lookback + pred_len:
                        return jsonify({'error': f'数据不足，回测需要 {lookback + pred_len} 条，当前只有 {len(df)} 条。如需预测未来请使用下方"预测未来"功能'}), 400
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = f"回测验证：前 {lookback} 条数据预测，第 {lookback+1}~{lookback+pred_len} 条真实数据对比"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_dt is not None:
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            actual_df = time_range_df.iloc[lookback:lookback+pred_len]
        else:
            actual_df = df.iloc[lookback:lookback+pred_len]

        for _, row in actual_df.iterrows():
            actual_data.append({
                'timestamp': row['timestamps'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Create chart - pass historical data start position
        if start_dt is not None:
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            historical_start_idx = 0
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_dt is not None:
                # Custom time period: use selected window data to calculate timestamps
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].dropna().diff().median()
                    if time_diff >= pd.Timedelta(days=1):
                        from pandas.tseries.offsets import BDay
                        future_timestamps = pd.bdate_range(start=last_timestamp + BDay(1), periods=pred_len)
                    else:
                        future_timestamps = pd.date_range(start=last_timestamp + time_diff, periods=pred_len, freq=time_diff)
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].dropna().diff().median()
                if time_diff >= pd.Timedelta(days=1):
                    from pandas.tseries.offsets import BDay
                    future_timestamps = pd.bdate_range(start=last_timestamp + BDay(1), periods=pred_len)
                else:
                    future_timestamps = pd.date_range(start=last_timestamp + time_diff, periods=pred_len, freq=time_diff)
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date_raw if start_date_raw else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ---------------------------------------------------------------------------
# Future prediction chart helper
# ---------------------------------------------------------------------------
def create_future_prediction_chart(historical_df, pred_df, future_timestamps, cutoff_date, label=''):
    """Chart for predict-future: historical candlestick + vertical cutoff line + predicted overlay."""
    from plotly.subplots import make_subplots

    hist_ts = historical_df['timestamps']
    kline_title = f'{label} K线图' if label else 'K线图'
    title = f'{label} 未来价格预测（数据截止 {cutoff_date.strftime("%Y-%m-%d")}）' if label else f'未来价格预测（数据截止 {cutoff_date.strftime("%Y-%m-%d")}）'

    # 检测时间精度，统一转字符串作为 category 轴标签（消除非交易日空白）
    _sample = hist_ts.iloc[0] if len(hist_ts) > 0 else pd.Timestamp(future_timestamps[0])
    _has_time = (pd.Timestamp(_sample).hour != 0 or pd.Timestamp(_sample).minute != 0)
    _ts_fmt = '%Y-%m-%d %H:%M' if _has_time else '%Y-%m-%d'
    hist_ts_str   = hist_ts.dt.strftime(_ts_fmt)
    future_ts_str = [pd.Timestamp(t).strftime(_ts_fmt) for t in future_timestamps]
    cutoff_str    = pd.Timestamp(cutoff_date).strftime(_ts_fmt)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(title, '成交量预测', kline_title),
        vertical_spacing=0.07,
        row_heights=[0.45, 0.25, 0.30]
    )

    # --- Historical close line ---
    fig.add_trace(go.Scatter(
        x=hist_ts_str,
        y=historical_df['close'],
        mode='lines',
        name='历史收盘价',
        line=dict(color='#1f77b4', width=2),
    ), row=1, col=1)

    # --- Predicted close line (dashed) ---
    fig.add_trace(go.Scatter(
        x=future_ts_str,
        y=pred_df['close'].tolist(),
        mode='lines',
        name='预测收盘价',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
    ), row=1, col=1)

    # --- Historical volume ---
    if 'volume' in historical_df.columns:
        fig.add_trace(go.Bar(
            x=hist_ts_str, y=historical_df['volume'],
            name='历史成交量', marker_color='#1f77b4', opacity=0.7,
        ), row=2, col=1)

    # --- Predicted volume ---
    if 'volume' in pred_df.columns:
        fig.add_trace(go.Bar(
            x=future_ts_str, y=pred_df['volume'].tolist(),
            name='预测成交量', marker_color='#ff7f0e', opacity=0.7,
        ), row=2, col=1)

    # --- Historical candlestick (row 3) ---
    fig.add_trace(go.Candlestick(
        x=hist_ts_str,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='历史K线',
        increasing_line_color='#FF4D4F',
        increasing_fillcolor='#FF4D4F',
        decreasing_line_color='#52C41A',
        decreasing_fillcolor='#52C41A',
        whiskerwidth=0.2,
        opacity=0.9,
    ), row=3, col=1)

    # --- Predicted candlestick (semi-transparent, row 3) ---
    if {'open', 'high', 'low', 'close'}.issubset(set(pred_df.columns)):
        fig.add_trace(go.Candlestick(
            x=future_ts_str,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='预测K线',
            increasing_line_color='#FF4D4F',
            increasing_fillcolor='rgba(255,77,79,0.4)',
            decreasing_line_color='#52C41A',
            decreasing_fillcolor='rgba(82,196,26,0.4)',
            whiskerwidth=0.2,
            opacity=0.7,
        ), row=3, col=1)

    # --- Vertical cutoff line (all rows) ---
    fig.add_vline(
        x=cutoff_str,
        line_width=2, line_dash='solid', line_color='#FFD700',
    )
    fig.add_annotation(
        x=cutoff_str,
        y=1, yref='paper',
        text='◀ 历史  预测 ▶',
        showarrow=False,
        font=dict(color='#FFD700', size=12),
        bgcolor='rgba(0,0,0,0.5)',
        xanchor='center',
    )

    fig.update_layout(
        title='价格/成交量/日K 预测结果',
        template='plotly_dark',
        paper_bgcolor='#0b1220',
        plot_bgcolor='#0b1220',
        height=1000,
        width=1400,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=12, color='#e5e7eb'),
        title_font_size=16,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    # 所有子图 x 轴使用 category 类型，消除周末/节假日空白
    fig.update_xaxes(
        type='category',
        rangeslider_visible=False,
        nticks=12,
    )
    fig.update_yaxes(title_text='价格 (元)', row=1, col=1)
    fig.update_yaxes(title_text='成交量',   row=2, col=1)
    fig.update_yaxes(title_text='价格 (元)', row=3, col=1)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ---------------------------------------------------------------------------
# /api/predict-future  ── predict the future using latest rows of a CSV
# ---------------------------------------------------------------------------
@app.route('/api/predict-future', methods=['POST'])
@login_required
def predict_future():
    """Predict future prices using the most recent `lookback` rows of a CSV file."""
    try:
        data = request.get_json()
        file_path   = data.get('file_path')
        lookback    = int(data.get('lookback', 400))
        pred_len    = int(data.get('pred_len', 30))
        temperature = float(data.get('temperature', 1.0))
        top_p       = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        if not file_path:
            return jsonify({'error': '文件路径不能为空'}), 400

        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400

        if len(df) < lookback:
            return jsonify({'error': f'数据不足，至少需要 {lookback} 条，当前只有 {len(df)} 条'}), 400

        if not (MODEL_AVAILABLE and predictor is not None):
            return jsonify({'error': 'Kronos 模型未加载，请先在上方加载模型'}), 400

        # --- Prepare input: most recent lookback rows ---
        required_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            required_cols.append('volume')

        historical_df = df.iloc[-lookback:].copy()
        x_df          = historical_df[required_cols]
        x_timestamp   = pd.Series(historical_df['timestamps'].values, name='timestamps')

        # --- Generate future timestamps ---
        from pandas.tseries.offsets import BDay
        cutoff_date = df['timestamps'].iloc[-1]
        time_diff   = df['timestamps'].dropna().diff().median()

        if time_diff >= pd.Timedelta(days=1):
            future_timestamps = pd.bdate_range(start=cutoff_date + BDay(1), periods=pred_len)
        else:
            future_timestamps = pd.date_range(start=cutoff_date + time_diff, periods=pred_len, freq=time_diff)

        y_timestamp = pd.Series(future_timestamps, name='timestamps')

        # --- Call Kronos model ---
        try:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
            )
        except Exception as e:
            return jsonify({'error': f'Kronos 模型预测失败: {str(e)}'}), 500

        # --- Build chart ---
        label = os.path.splitext(os.path.basename(file_path))[0]
        chart_json = create_future_prediction_chart(historical_df, pred_df, future_timestamps, cutoff_date, label)

        # --- Build prediction list ---
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f'T+{i}',
                'open':   float(row['open']),
                'high':   float(row['high']),
                'low':    float(row['low']),
                'close':  float(row['close']),
                'volume': float(row.get('volume', 0.0)),
            })

        return jsonify({
            'success': True,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'cutoff_date':  cutoff_date.strftime('%Y-%m-%d'),
            'pred_start':   future_timestamps[0].strftime('%Y-%m-%d'),
            'pred_end':     future_timestamps[-1].strftime('%Y-%m-%d'),
            'used_rows':    lookback,
            'message': (f'预测完成：基于截至 {cutoff_date.strftime("%Y-%m-%d")} 的最近 {lookback} 条数据，'
                        f'预测 {future_timestamps[0].strftime("%Y-%m-%d")} 至 {future_timestamps[-1].strftime("%Y-%m-%d")} '
                        f'共 {pred_len} 个时间单位'),
        })

    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500


@app.route('/api/load-model', methods=['POST'])
@login_required
def load_model():
    """Load Kronos model with retry mechanism"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Try to load with retry mechanism
        import time
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to load model (attempt {attempt + 1}/{max_retries})...")
                
                # Load tokenizer and model with timeout
                print(f"Loading tokenizer from: {model_config['tokenizer_id']}")
                print(f"Using mirror: {os.environ.get('HF_ENDPOINT', 'default')}")
                
                tokenizer = KronosTokenizer.from_pretrained(
                    model_config['tokenizer_id'],
                    local_files_only=False,
                    resume_download=True
                )
                
                print(f"Loading model from: {model_config['model_id']}")
                model = Kronos.from_pretrained(
                    model_config['model_id'],
                    local_files_only=False,
                    resume_download=True
                )
                
                # Create predictor
                predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
                
                return jsonify({
                    'success': True,
                    'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
                    'model_info': {
                        'name': model_config['name'],
                        'params': model_config['params'],
                        'context_length': model_config['context_length'],
                        'description': model_config['description']
                    }
                })
                
            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed, provide helpful error message
                    error_details = f'\n\n错误详情: {error_msg}\n\n'
                    
                    if "SSL" in error_msg or "HTTPSConnectionPool" in error_msg or "Connection" in error_msg or "cannot find the requested files" in error_msg:
                        suggestion = (
                            '🔄 已配置使用国内镜像站点 (hf-mirror.com)\n\n'
                            '解决方案:\n'
                            '1. 服务器已重启生效，请再次尝试加载模型\n'
                            '2. 确保网络连接正常，可以访问 hf-mirror.com\n'
                            '3. 首次下载模型较大，请耐心等待\n'
                            '4. 如持续失败，可手动下载模型\n\n'
                            '手动下载方法:\n'
                            f'访问: https://hf-mirror.com/NeoQuasar/{model_config["name"]}\n'
                            '下载所有文件到本地，然后使用本地路径加载'
                        )
                    else:
                        suggestion = (
                            '建议:\n'
                            '1. 检查错误详情\n'
                            '2. 确保有足够的磁盘空间\n'
                            '3. 尝试其他设备类型（CPU/CUDA）\n'
                            '4. 查看控制台完整错误日志'
                        )
                    
                    return jsonify({
                        'error': f'模型加载失败{error_details}{suggestion}'
                    }), 500
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
@login_required
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
@login_required
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': 'Kronos',
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

def _mask(s: str) -> str:
    return s[:4] + '****' + s[-4:] if len(s) > 8 else '****'

@app.route('/api/tushare-token', methods=['GET'])
@login_required
def get_tushare_token():
    token = TUSHARE_TOKEN
    if token:
        return jsonify({'configured': True, 'masked': _mask(token)})
    return jsonify({'configured': False, 'masked': ''})

@app.route('/api/tushare-token', methods=['POST'])
@login_required
def set_tushare_token():
    global TUSHARE_TOKEN
    body = request.get_json() or {}
    token = body.get('token', '').strip()
    if not token:
        return jsonify({'error': 'Token 不能为空'}), 400
    TUSHARE_TOKEN = token
    cfg = _load_config()
    cfg['tushare_token'] = token
    _save_config(cfg)
    return jsonify({'success': True, 'message': 'Tushare Token 保存成功'})

@app.route('/api/tqsdk-credentials', methods=['GET'])
@login_required
def get_tqsdk_credentials():
    return jsonify({
        'configured': bool(TQSDK_USERNAME and TQSDK_PASSWORD),
        'username': TQSDK_USERNAME,
    })

@app.route('/api/tqsdk-credentials', methods=['POST'])
@login_required
def set_tqsdk_credentials():
    global TQSDK_USERNAME, TQSDK_PASSWORD
    body = request.get_json() or {}
    username = body.get('username', '').strip()
    password = body.get('password', '').strip()
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400
    TQSDK_USERNAME = username
    TQSDK_PASSWORD = password
    cfg = _load_config()
    cfg['tqsdk_username'] = username
    cfg['tqsdk_password'] = password
    _save_config(cfg)
    return jsonify({'success': True, 'message': 'TQSdk 账号保存成功'})

@app.route('/api/help')
@login_required
def get_help():
    """返回使用教程 Markdown 文本。"""
    doc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '使用教程.md')
    if not os.path.exists(doc_path):
        return jsonify({'error': '教程文件不存在'}), 404
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify({'content': content})

@app.route('/api/refresh-symbols', methods=['POST'])
@login_required
def api_refresh_symbols():
    """从 TQSdk 拉取全市场期货品种列表，更新本地缓存。"""
    count, err = _refresh_symbols_from_tqsdk()
    if err:
        return jsonify({'error': err}), 400
    # 返回缓存更新时间
    updated_at = ''
    try:
        with open(_SYMBOLS_CACHE_PATH, 'r', encoding='utf-8') as f:
            updated_at = json.load(f).get('updated_at', '')
    except Exception:
        pass
    return jsonify({'success': True, 'count': count,
                    'message': f'已同步 {count} 个期货品种', 'updated_at': updated_at})


@app.route('/api/search-symbols')
@login_required
def api_search_symbols():
    """期货品种搜索接口，支持代码/中文名/交易所模糊匹配。"""
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    q_lower = q.lower()
    results = []
    seen = set()
    for product, exchange in _PRODUCT_EXCHANGE.items():
        if product in seen:
            continue
        name = (_PRODUCT_NAME.get(product)
                or _PRODUCT_NAME.get(product.upper())
                or _PRODUCT_NAME.get(product.lower()) or '')
        if (q_lower in product.lower()
                or q_lower in name
                or q_lower in exchange.lower()):
            seen.add(product)
            results.append({'code': product, 'name': name, 'exchange': exchange})
    # 排序：代码前缀匹配 > 代码包含 > 名称前缀 > 其他
    results.sort(key=lambda x: (
        0 if x['code'].lower().startswith(q_lower) else
        1 if q_lower in x['code'].lower() else
        2 if x['name'].startswith(q) else 3
    ))
    return jsonify(results[:25])


@app.route('/api/symbols-cache-info')
@login_required
def api_symbols_cache_info():
    """返回品种库缓存状态（更新时间、品种数量）。"""
    if not os.path.exists(_SYMBOLS_CACHE_PATH):
        return jsonify({'exists': False})
    try:
        with open(_SYMBOLS_CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        return jsonify({
            'exists': True,
            'count': len(cache.get('exchange', {})),
            'updated_at': cache.get('updated_at', ''),
        })
    except Exception as e:
        return jsonify({'exists': False, 'error': str(e)})


@app.route('/api/sysinfo')
@login_required
def sysinfo():
    """Return Python executable and tqsdk availability for diagnostics."""
    import subprocess, sys as _sys
    info = {
        'flask_executable': _sys.executable,
        'tqsdk_executable': _TQSDK_PYTHON,
        'python_version': _sys.version,
    }
    # Check tqsdk in the Flask venv
    r = subprocess.run([_sys.executable, '-c', 'import tqsdk; print(tqsdk.__version__)'],
                       capture_output=True, text=True, timeout=20)
    info['tqsdk_in_flask_env'] = r.stdout.strip() if r.returncode == 0 else f'NOT FOUND: {r.stderr.strip()[:200]}'
    # Check tqsdk in the chosen tqsdk interpreter
    if _TQSDK_PYTHON != _sys.executable:
        r2 = subprocess.run([_TQSDK_PYTHON, '-c', 'import tqsdk; print(tqsdk.__version__)'],
                            capture_output=True, text=True, timeout=20)
        info['tqsdk_in_tqsdk_env'] = r2.stdout.strip() if r2.returncode == 0 else f'NOT FOUND: {r2.stderr.strip()[:200]}'
    return jsonify(info)

@app.route('/api/test-tqsdk', methods=['POST'])
@login_required
def test_tqsdk():
    """Debug endpoint using subprocess approach."""
    global TQSDK_USERNAME, TQSDK_PASSWORD
    body = request.get_json() or {}
    symbol = body.get('symbol', 'KQ.m@SHFE.rb')

    if not TQSDK_USERNAME or not TQSDK_PASSWORD:
        return jsonify({'error': 'TQSdk 账号未配置'}), 400

    import subprocess, tempfile, sys as _sys, os

    script = (
        'import sys, warnings; warnings.filterwarnings("ignore")\n'
        'import asyncio, time, pandas as pd\n'
        f'username = {repr(TQSDK_USERNAME)}\n'
        f'password = {repr(TQSDK_PASSWORD)}\n'
        f'symbol   = {repr(symbol)}\n'
        'out_csv  = sys.argv[1]\n'
        'loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)\n'
        'try:\n'
        '    from tqsdk import TqApi, TqAuth\n'
        '    api = TqApi(auth=TqAuth(username, password))\n'
        '    klines = api.get_kline_serial(symbol, 86400, data_length=5)\n'
        '    deadline = time.time() + 30\n'
        '    while not api.is_serial_ready(klines):\n'
        '        if not api.wait_update(deadline=deadline):\n'
        '            sys.stderr.write("TIMEOUT"); sys.exit(1)\n'
        '    pd.DataFrame(klines).to_csv(out_csv, index=False)\n'
        '    api.close()\n'
        'except Exception as e:\n'
        '    import traceback; sys.stderr.write(traceback.format_exc()); sys.exit(1)\n'
        'finally:\n'
        '    loop.close()\n'
    )

    tmp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_csv.close()
    tmp_py = tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8')
    tmp_py.write(script)
    tmp_py.close()

    try:
        proc = subprocess.run(
            [_TQSDK_PYTHON, tmp_py.name, tmp_csv.name],
            capture_output=True, text=True, timeout=40
        )
        stderr_out = proc.stderr.strip()
        if proc.returncode != 0:
            return jsonify({'error': stderr_out or 'subprocess failed'}), 500
        df = pd.read_csv(tmp_csv.name)
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': list(df.columns),
            'sample': df.tail(2).to_dict(orient='records'),
            'stderr': stderr_out,
        })
    except subprocess.TimeoutExpired:
        return jsonify({'error': '子进程超时（40s）'}), 500
    finally:
        os.unlink(tmp_py.name)
        try: os.unlink(tmp_csv.name)
        except: pass

@app.route('/api/futures-symbols', methods=['GET'])
@login_required
def get_futures_symbols():
    """返回当前自定义品种补充列表（futures_symbols.json 的内容）。"""
    if os.path.exists(_SYMBOLS_CACHE_PATH):
        try:
            with open(_SYMBOLS_CACHE_PATH, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            return jsonify({'success': True, 'symbols': cache})
        except Exception as e:
            return jsonify({'error': f'读取失败: {e}'}), 500
    return jsonify({'success': True, 'symbols': {'exchange': {}, 'name': {}}})


@app.route('/api/futures-symbols', methods=['POST'])
@login_required
def add_futures_symbol():
    """新增一个自定义期货品种，写入 futures_symbols.json 并立即生效。"""
    data = request.get_json() or {}
    product = (data.get('product') or '').strip()
    exchange = (data.get('exchange') or '').strip().upper()
    name     = (data.get('name') or '').strip()

    if not product or not exchange:
        return jsonify({'error': '品种代码和交易所不能为空'}), 400
    if exchange not in {'SHFE', 'DCE', 'CZCE', 'CFFEX', 'INE', 'GFEX'}:
        return jsonify({'error': f'不支持的交易所: {exchange}'}), 400

    # 规范化大小写
    product_key = product.upper() if exchange in ('CZCE', 'CFFEX') else product.lower()

    # 读取现有缓存
    cache = {'exchange': {}, 'name': {}}
    if os.path.exists(_SYMBOLS_CACHE_PATH):
        try:
            with open(_SYMBOLS_CACHE_PATH, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            pass

    cache.setdefault('exchange', {})[product_key] = exchange
    if name:
        cache.setdefault('name', {})[product_key] = name

    # 写入文件
    try:
        with open(_SYMBOLS_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return jsonify({'error': f'保存失败: {e}'}), 500

    # 立即更新内存映射
    _PRODUCT_EXCHANGE[product_key] = exchange
    if name:
        _PRODUCT_NAME[product_key] = name

    return jsonify({
        'success': True,
        'message': f'已添加 {product_key} → {exchange}（{name or "无中文名"}），立即生效'
    })


@app.route('/api/futures-symbols/<product>', methods=['DELETE'])
@login_required
def delete_futures_symbol(product):
    """从自定义列表中删除一个品种。"""
    if not os.path.exists(_SYMBOLS_CACHE_PATH):
        return jsonify({'error': '自定义列表为空'}), 404
    try:
        with open(_SYMBOLS_CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except Exception as e:
        return jsonify({'error': f'读取失败: {e}'}), 500

    removed = False
    for key in list(cache.get('exchange', {}).keys()):
        if key.lower() == product.lower():
            cache['exchange'].pop(key, None)
            cache.get('name', {}).pop(key, None)
            _PRODUCT_EXCHANGE.pop(key, None)
            _PRODUCT_NAME.pop(key, None)
            removed = True

    if not removed:
        return jsonify({'error': f'未找到自定义品种: {product}'}), 404

    with open(_SYMBOLS_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    return jsonify({'success': True, 'message': f'已删除 {product}'})


if __name__ == '__main__':
    import signal, logging

    # 支持通过环境变量覆盖端口，方便云服务器部署
    _port = int(os.environ.get('KRONOS_PORT', 7070))

    # 结构化日志（systemd 下 journalctl 可直接采集）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    logger = logging.getLogger('kronos')

    logger.info("Kronos Web UI 启动中...")
    logger.info(f"模型可用: {MODEL_AVAILABLE}")
    logger.info(f"监听端口: {_port}")

    # 优雅退出：收到 SIGTERM（systemd stop）时正常结束
    def _handle_sigterm(signum, frame):
        logger.info("收到 SIGTERM，正在退出...")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # use_reloader=False 避免 _find_tqsdk_python() 在子进程中重复执行
    app.run(debug=False, host='0.0.0.0', port=_port,
            use_reloader=False, threaded=True)
