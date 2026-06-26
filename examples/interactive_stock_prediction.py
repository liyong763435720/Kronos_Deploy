#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式股票预测程序
基于Kronos模型和Akshare数据源的股票预测系统
支持用户输入股票代码或上传TXT文件，预测未来一个月的股票走势
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import warnings
from datetime import datetime, timedelta
import akshare as ak
import time
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 添加模型路径
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


class InteractiveStockPredictor:
    """交互式股票预测器"""
    
    def __init__(self):
        self.predictor = None
        self.model_loaded = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
    
    def load_model(self):
        """加载Kronos模型"""
        try:
            print("正在加载Kronos模型...")
            print("这可能需要几分钟时间，请耐心等待...")
            
            # 加载tokenizer和模型
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
            
            # 创建预测器
            self.predictor = KronosPredictor(
                model=model, 
                tokenizer=tokenizer, 
                device=self.device, 
                max_context=512
            )
            
            self.model_loaded = True
            print("✅ Kronos模型加载成功!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("请检查网络连接和模型文件")
            return False
    
    def get_trading_days(self, start_date, end_date):
        """获取交易日历（排除周末）"""
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            # 排除周末 (0=Monday, 6=Sunday)
            if current_date.weekday() < 5:
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def download_stock_data(self, stock_code, days_back=100, days_forward=30):
        """下载股票数据"""
        try:
            print(f"正在下载股票 {stock_code} 的数据...")
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back * 2)  # 多取一些天数确保有足够的交易日
            
            # 获取历史数据
            print(f"下载时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
            
            # 使用akshare下载数据
            data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust="qfq"  # 前复权
            )
            
            if data.empty:
                raise ValueError(f"未找到股票 {stock_code} 的数据")
            
            # 重命名列以匹配标准格式
            data = data.rename(columns={
                '日期': 'Date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            
            # 设置日期为索引
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            
            # 按日期排序
            data = data.sort_index()
            
            # 选择最近100个交易日的数据
            if len(data) > days_back:
                data = data.tail(days_back)
            
            print(f"✅ 成功下载 {len(data)} 个交易日的数据")
            print(f"数据范围: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}")
            
            return data
            
        except Exception as e:
            print(f"❌ 下载股票数据失败: {str(e)}")
            return None
    
    def prepare_prediction_data(self, stock_data, days_forward=30):
        """准备预测数据"""
        try:
            # 确保数据包含必要的列
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in required_cols:
                if col not in stock_data.columns:
                    if col == 'amount' and 'volume' in stock_data.columns:
                        # 如果没有成交额，用成交量*均价估算
                        stock_data['amount'] = stock_data['volume'] * stock_data[['open', 'high', 'low', 'close']].mean(axis=1)
                    else:
                        stock_data[col] = 0.0
            
            # 创建时间戳
            x_timestamp = stock_data.index
            y_timestamp = pd.date_range(
                start=stock_data.index[-1] + timedelta(days=1),
                periods=days_forward,
                freq='D'
            )
            
            # 过滤掉周末的预测日期
            y_timestamp = [d for d in y_timestamp if d.weekday() < 5]
            y_timestamp = pd.DatetimeIndex(y_timestamp)
            
            print(f"历史数据: {len(stock_data)} 个交易日")
            print(f"预测期间: {len(y_timestamp)} 个交易日")
            
            return stock_data, x_timestamp, y_timestamp
            
        except Exception as e:
            print(f"❌ 准备预测数据失败: {str(e)}")
            return None, None, None
    
    def make_prediction(self, stock_data, x_timestamp, y_timestamp):
        """使用Kronos模型进行预测"""
        if not self.model_loaded:
            print("❌ 模型未加载，请先加载模型")
            return None
        
        try:
            print("正在进行股票预测...")
            print("这可能需要几分钟时间，请耐心等待...")
            
            # 使用Kronos进行预测
            pred_df = self.predictor.predict(
                df=stock_data,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=len(y_timestamp),
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=True
            )
            
            print("✅ 预测完成!")
            return pred_df
            
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            return None
    
    def plot_prediction_results(self, stock_data, pred_df, stock_code):
        """绘制预测结果"""
        try:
            # 合并历史数据和预测数据
            historical_data = stock_data[['open', 'high', 'low', 'close', 'volume']].copy()
            historical_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            pred_data = pred_df[['open', 'high', 'low', 'close', 'volume']].copy()
            pred_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # 绘制价格图
            ax1.plot(historical_data.index, historical_data['Close'], 
                    label='历史收盘价', color='blue', linewidth=2)
            ax1.plot(pred_data.index, pred_data['Close'], 
                    label='预测收盘价', color='red', linewidth=2, linestyle='--')
            
            ax1.set_title(f'股票 {stock_code} 价格预测', fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格 (元)', fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 绘制成交量图
            ax2.bar(historical_data.index, historical_data['Volume'], 
                   label='历史成交量', color='blue', alpha=0.7, width=0.8)
            ax2.bar(pred_data.index, pred_data['Volume'], 
                   label='预测成交量', color='red', alpha=0.7, width=0.8)
            
            ax2.set_title('成交量预测', fontsize=14, fontweight='bold')
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.set_xlabel('日期', fontsize=12)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图片
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'prediction_{stock_code}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"预测结果图表已保存: {filename}")
            
            plt.show()
            
            return filename
            
        except Exception as e:
            print(f"❌ 绘制图表失败: {str(e)}")
            return None
    
    def save_prediction_results(self, stock_data, pred_df, stock_code):
        """保存预测结果到文件"""
        try:
            # 合并数据
            historical_data = stock_data.copy()
            historical_data['type'] = 'historical'
            
            pred_data = pred_df.copy()
            pred_data['type'] = 'prediction'
            
            combined_data = pd.concat([historical_data, pred_data])
            
            # 保存到CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'prediction_results_{stock_code}_{timestamp}.csv'
            combined_data.to_csv(filename, encoding='utf-8-sig')
            
            print(f"预测结果已保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ 保存结果失败: {str(e)}")
            return None
    
    def get_stock_codes_from_user(self):
        """获取用户输入的股票代码"""
        print("\n" + "="*60)
        print("股票预测系统")
        print("="*60)
        print("请选择输入方式:")
        print("1. 手动输入股票代码")
        print("2. 从TXT文件读取股票代码")
        print()
        
        while True:
            choice = input("请选择 (1/2): ").strip()
            
            if choice == '1':
                return self.get_manual_stock_codes()
            elif choice == '2':
                return self.get_stock_codes_from_file()
            else:
                print("❌ 请输入 1 或 2")
    
    def get_manual_stock_codes(self):
        """手动输入股票代码"""
        print("\n手动输入股票代码")
        print("-" * 30)
        print("格式说明:")
        print("- 多个股票代码用逗号分隔")
        print("- 支持A股代码格式: 600030, 002261, 688326, 300364")
        print("- 示例: 600030,002261")
        print()
        
        while True:
            stock_input = input("请输入股票代码: ").strip()
            if not stock_input:
                print("❌ 请输入有效的股票代码")
                continue
            
            # 解析股票代码
            stock_codes = [code.strip() for code in stock_input.split(',')]
            stock_codes = [code for code in stock_codes if code]
            
            if not stock_codes:
                print("❌ 请输入有效的股票代码")
                continue
            
            # 验证股票代码格式
            valid_codes = []
            for code in stock_codes:
                if code.isdigit() and len(code) == 6:
                    valid_codes.append(code)
                else:
                    print(f"⚠️  股票代码 {code} 格式不正确，已跳过")
            
            if not valid_codes:
                print("❌ 没有有效的股票代码")
                continue
            
            return valid_codes
    
    def get_stock_codes_from_file(self):
        """从TXT文件读取股票代码"""
        print("\n从TXT文件读取股票代码")
        print("-" * 30)
        print("文件格式说明:")
        print("- 每行一个股票代码")
        print("- 支持注释行（以#开头）")
        print("- 示例文件内容:")
        print("  # 这是注释行")
        print("  600030")
        print("  002261")
        print()
        
        while True:
            filename = input("请输入TXT文件名 (例如: stock_codes.txt): ").strip()
            if not filename:
                print("❌ 请输入文件名")
                continue
            
            # 如果用户没有输入扩展名，自动添加.txt
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                stock_codes = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        stock_codes.append(line)
                
                if not stock_codes:
                    print("❌ 文件中没有有效的股票代码")
                    continue
                
                # 验证股票代码格式
                valid_codes = []
                for code in stock_codes:
                    if code.isdigit() and len(code) == 6:
                        valid_codes.append(code)
                    else:
                        print(f"⚠️  股票代码 {code} 格式不正确，已跳过")
                
                if not valid_codes:
                    print("❌ 文件中没有有效的股票代码")
                    continue
                
                return valid_codes
                
            except FileNotFoundError:
                print(f"❌ 文件 {filename} 不存在")
                continue
            except Exception as e:
                print(f"❌ 读取文件失败: {e}")
                continue
    
    def run_prediction_for_stock(self, stock_code):
        """对单个股票进行预测"""
        print(f"\n开始预测股票: {stock_code}")
        print("-" * 50)
        
        # 1. 下载股票数据
        stock_data = self.download_stock_data(stock_code, days_back=100, days_forward=30)
        if stock_data is None:
            return False
        
        # 2. 准备预测数据
        stock_data, x_timestamp, y_timestamp = self.prepare_prediction_data(stock_data, days_forward=30)
        if stock_data is None:
            return False
        
        # 3. 进行预测
        pred_df = self.make_prediction(stock_data, x_timestamp, y_timestamp)
        if pred_df is None:
            return False
        
        # 4. 显示预测结果
        print(f"\n预测结果预览:")
        print(pred_df.head())
        
        # 5. 绘制图表
        chart_file = self.plot_prediction_results(stock_data, pred_df, stock_code)
        
        # 6. 保存结果
        result_file = self.save_prediction_results(stock_data, pred_df, stock_code)
        
        print(f"\n✅ 股票 {stock_code} 预测完成!")
        if chart_file:
            print(f"📊 图表文件: {chart_file}")
        if result_file:
            print(f"📄 数据文件: {result_file}")
        
        return True
    
    def run(self):
        """运行主程序"""
        print("欢迎使用Kronos股票预测系统!")
        print("本系统可以预测股票未来一个月的走势")
        print()
        
        # 1. 加载模型
        if not self.load_model():
            return
        
        # 2. 获取股票代码
        stock_codes = self.get_stock_codes_from_user()
        if not stock_codes:
            print("❌ 未获取到有效的股票代码")
            return
        
        print(f"\n将预测以下股票: {', '.join(stock_codes)}")
        
        # 3. 对每只股票进行预测
        success_count = 0
        for i, stock_code in enumerate(stock_codes, 1):
            print(f"\n[{i}/{len(stock_codes)}] 处理股票: {stock_code}")
            
            try:
                if self.run_prediction_for_stock(stock_code):
                    success_count += 1
                else:
                    print(f"❌ 股票 {stock_code} 预测失败")
            except Exception as e:
                print(f"❌ 股票 {stock_code} 处理出错: {str(e)}")
            
            # 添加延迟避免请求过于频繁
            if i < len(stock_codes):
                print("等待3秒后处理下一只股票...")
                time.sleep(3)
        
        # 4. 总结
        print(f"\n" + "="*60)
        print(f"预测完成! 成功预测 {success_count}/{len(stock_codes)} 只股票")
        print("="*60)


def main():
    """主函数"""
    try:
        predictor = InteractiveStockPredictor()
        predictor.run()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        print("请检查网络连接和依赖包是否正确安装")


if __name__ == "__main__":
    main()
