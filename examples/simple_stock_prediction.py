#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版股票预测程序
适用于没有GPU或模型加载有问题的环境
使用简单的技术分析方法进行预测
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class SimpleStockPredictor:
    """简化版股票预测器（基于技术分析）"""
    
    def __init__(self):
        self.device = "CPU"
        print(f"使用设备: {self.device}")
    
    def download_stock_data(self, stock_code, days_back=100):
        """下载股票数据"""
        try:
            print(f"正在下载股票 {stock_code} 的数据...")
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back * 2)
            
            print(f"下载时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
            
            # 使用akshare下载数据
            data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust="qfq"
            )
            
            if data.empty:
                raise ValueError(f"未找到股票 {stock_code} 的数据")
            
            # 重命名列
            data = data.rename(columns={
                '日期': 'Date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            data = data.sort_index()
            
            # 选择最近100个交易日
            if len(data) > days_back:
                data = data.tail(days_back)
            
            print(f"✅ 成功下载 {len(data)} 个交易日的数据")
            return data
            
        except Exception as e:
            print(f"❌ 下载股票数据失败: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        df = data.copy()
        
        # 移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        
        # RSI指标
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD指标
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # 布林带
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        return df
    
    def simple_prediction(self, data, days_forward=30):
        """简单的预测方法"""
        df = self.calculate_technical_indicators(data)
        
        # 获取最新数据
        latest = df.iloc[-1]
        
        # 基于技术指标的简单预测
        predictions = []
        current_price = latest['close']
        
        # 趋势判断
        if latest['MA5'] > latest['MA10'] > latest['MA20']:
            trend = "上升"
            trend_factor = 1.02
        elif latest['MA5'] < latest['MA10'] < latest['MA20']:
            trend = "下降"
            trend_factor = 0.98
        else:
            trend = "震荡"
            trend_factor = 1.0
        
        # RSI判断
        if latest['RSI'] > 70:
            rsi_factor = 0.99  # 超买
        elif latest['RSI'] < 30:
            rsi_factor = 1.01  # 超卖
        else:
            rsi_factor = 1.0
        
        # 生成预测
        for i in range(days_forward):
            # 添加随机波动
            random_factor = np.random.normal(1.0, 0.02)
            
            # 计算预测价格
            predicted_price = current_price * (trend_factor ** (i+1)) * (rsi_factor ** (i+1)) * random_factor
            
            # 生成OHLC数据
            high = predicted_price * (1 + abs(np.random.normal(0, 0.01)))
            low = predicted_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = predicted_price * (1 + np.random.normal(0, 0.005))
            volume = latest['volume'] * (1 + np.random.normal(0, 0.1))
            
            predictions.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': predicted_price,
                'volume': volume
            })
            
            current_price = predicted_price
        
        return pd.DataFrame(predictions)
    
    def plot_prediction_results(self, stock_data, pred_df, stock_code):
        """绘制预测结果"""
        try:
            # 创建图表
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            
            # 价格图
            ax1.plot(stock_data.index, stock_data['close'], 
                    label='历史收盘价', color='blue', linewidth=2)
            ax1.plot(pred_df.index, pred_df['close'], 
                    label='预测收盘价', color='red', linewidth=2, linestyle='--')
            ax1.set_title(f'股票 {stock_code} 价格预测 (技术分析)', fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格 (元)', fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 成交量图
            ax2.bar(stock_data.index, stock_data['volume'], 
                   label='历史成交量', color='blue', alpha=0.7, width=0.8)
            ax2.bar(pred_df.index, pred_df['volume'], 
                   label='预测成交量', color='red', alpha=0.7, width=0.8)
            ax2.set_title('成交量预测', fontsize=14, fontweight='bold')
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 技术指标图
            if 'MA5' in stock_data.columns:
                ax3.plot(stock_data.index, stock_data['MA5'], label='MA5', color='orange', linewidth=1)
                ax3.plot(stock_data.index, stock_data['MA20'], label='MA20', color='green', linewidth=1)
                ax3.plot(stock_data.index, stock_data['close'], label='收盘价', color='blue', linewidth=1)
                ax3.set_title('技术指标', fontsize=14, fontweight='bold')
                ax3.set_ylabel('价格', fontsize=12)
                ax3.legend(fontsize=12)
                ax3.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图片
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'simple_prediction_{stock_code}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"预测结果图表已保存: {filename}")
            
            plt.show()
            return filename
            
        except Exception as e:
            print(f"❌ 绘制图表失败: {str(e)}")
            return None
    
    def get_stock_codes_from_user(self):
        """获取用户输入的股票代码"""
        print("\n" + "="*60)
        print("简化版股票预测系统 (基于技术分析)")
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
            
            stock_codes = [code.strip() for code in stock_input.split(',')]
            stock_codes = [code for code in stock_codes if code]
            
            if not stock_codes:
                print("❌ 请输入有效的股票代码")
                continue
            
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
        
        while True:
            filename = input("请输入TXT文件名 (例如: stock_codes.txt): ").strip()
            if not filename:
                print("❌ 请输入文件名")
                continue
            
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
        stock_data = self.download_stock_data(stock_code, days_back=100)
        if stock_data is None:
            return False
        
        # 2. 进行预测
        print("正在进行技术分析预测...")
        pred_df = self.simple_prediction(stock_data, days_forward=30)
        
        # 创建预测日期索引（严格匹配预测长度，使用工作日频率）
        start_date = stock_data.index[-1] + timedelta(days=1)
        pred_dates = pd.bdate_range(start=start_date, periods=len(pred_df))
        pred_df.index = pred_dates
        
        print("✅ 预测完成!")
        
        # 3. 显示预测结果
        print(f"\n预测结果预览:")
        print(pred_df.head())
        
        # 4. 绘制图表
        chart_file = self.plot_prediction_results(stock_data, pred_df, stock_code)
        
        # 5. 保存结果
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'simple_prediction_results_{stock_code}_{timestamp}.csv'
            pred_df.to_csv(filename, encoding='utf-8-sig')
            print(f"预测结果已保存: {filename}")
        except Exception as e:
            print(f"保存结果失败: {e}")
        
        print(f"\n✅ 股票 {stock_code} 预测完成!")
        return True
    
    def run(self):
        """运行主程序"""
        print("欢迎使用简化版股票预测系统!")
        print("本系统基于技术分析方法进行预测")
        print("注意: 这是简化版本，预测结果仅供参考")
        print()
        
        # 获取股票代码
        stock_codes = self.get_stock_codes_from_user()
        if not stock_codes:
            print("❌ 未获取到有效的股票代码")
            return
        
        print(f"\n将预测以下股票: {', '.join(stock_codes)}")
        
        # 对每只股票进行预测
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
            
            if i < len(stock_codes):
                print("等待2秒后处理下一只股票...")
                time.sleep(2)
        
        print(f"\n" + "="*60)
        print(f"预测完成! 成功预测 {success_count}/{len(stock_codes)} 只股票")
        print("="*60)


def main():
    """主函数"""
    try:
        predictor = SimpleStockPredictor()
        predictor.run()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()
