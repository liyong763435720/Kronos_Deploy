#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据合并和财务指标分析脚本
合并多只股票的交易数据并下载财务指标
"""

import akshare as ak
import pandas as pd
import os
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

def download_and_merge_stock_data(stock_codes, start_date, end_date, output_dir="combined_data"):
    """
    下载并合并多只股票的交易数据
    
    Args:
        stock_codes (list): 股票代码列表
        start_date (str): 开始日期，格式：'YYYY-MM-DD'
        end_date (str): 结束日期，格式：'YYYY-MM-DD'
        output_dir (str): 输出目录
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 转换日期格式
    start_date_formatted = start_date.replace('-', '')
    end_date_formatted = end_date.replace('-', '')
    
    print(f"开始下载并合并A股数据...")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"股票代码: {stock_codes}")
    print("-" * 60)
    
    all_data = []
    success_count = 0
    failed_stocks = []
    
    for i, code in enumerate(stock_codes, 1):
        try:
            print(f"[{i}/{len(stock_codes)}] 正在下载 {code}...")
            
            # 使用akshare下载数据
            data = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date_formatted,
                end_date=end_date_formatted,
                adjust="qfq"  # 前复权
            )
            
            if data.empty:
                print(f"  ❌ {code}: 未找到数据")
                failed_stocks.append(code)
                continue
            
            # 重命名列以匹配标准格式
            data = data.rename(columns={
                '日期': 'Date',
                '开盘': 'Open',
                '收盘': 'Close',
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume',
                '成交额': 'Amount',
                '振幅': 'Amplitude',
                '涨跌幅': 'Change',
                '涨跌额': 'ChangeAmount',
                '换手率': 'Turnover'
            })
            
            # 设置日期为索引
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            
            # 添加股票代码列
            data['Stock_Code'] = code
            
            # 重新排列列的顺序
            cols = ['Stock_Code'] + [col for col in data.columns if col != 'Stock_Code']
            data = data[cols]
            
            all_data.append(data)
            
            print(f"  ✅ {code}: 成功下载 {len(data)} 条记录")
            print(f"     数据范围: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}")
            
            success_count += 1
            
            # 添加延迟避免请求过于频繁
            if i < len(stock_codes):
                time.sleep(2)
            
        except Exception as e:
            print(f"  ❌ {code}: 下载失败 - {str(e)}")
            failed_stocks.append(code)
        
        print()
    
    if all_data:
        # 合并所有数据
        print("正在合并数据...")
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data.sort_values(['Stock_Code', 'Date'])
        
        # 保存合并后的数据
        filename = "combined_stock_data.csv"
        filepath = os.path.join(output_dir, filename)
        combined_data.to_csv(filepath, encoding='utf-8-sig')
        
        print(f"✅ 合并完成!")
        print(f"   总记录数: {len(combined_data)}")
        print(f"   股票数量: {len(combined_data['Stock_Code'].unique())}")
        print(f"   保存至: {filepath}")
    
    # 输出总结
    print("=" * 60)
    print("数据下载和合并完成!")
    print(f"成功下载: {success_count}/{len(stock_codes)} 支股票")
    
    if failed_stocks:
        print(f"失败的股票: {', '.join(failed_stocks)}")
    
    print(f"数据保存在: {os.path.abspath(output_dir)}")
    
    return combined_data if all_data else None

def download_quarterly_financial_data(stock_codes, output_dir="combined_data"):
    """
    下载季度财务数据
    
    Args:
        stock_codes (list): 股票代码列表
        output_dir (str): 输出目录
    """
    
    print("\n" + "=" * 60)
    print("开始下载季度财务数据...")
    print("-" * 60)
    
    all_quarterly_data = []
    success_count = 0
    failed_stocks = []
    
    for i, code in enumerate(stock_codes, 1):
        try:
            print(f"[{i}/{len(stock_codes)}] 正在下载 {code} 的季度财务数据...")
            
            # 获取财务数据
            financial_data = ak.stock_financial_abstract(symbol=code)
            
            if financial_data.empty:
                print(f"  ❌ {code}: 未找到季度财务数据")
                failed_stocks.append(code)
                continue
            
            # 添加股票代码
            financial_data['股票代码'] = code
            financial_data['更新时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            all_quarterly_data.append(financial_data)
            
            print(f"  ✅ {code}: 成功获取季度财务数据 ({len(financial_data)} 条记录)")
            
            success_count += 1
            
            # 添加延迟避免请求过于频繁
            if i < len(stock_codes):
                time.sleep(3)
            
        except Exception as e:
            print(f"  ❌ {code}: 季度财务数据下载失败 - {str(e)}")
            failed_stocks.append(code)
        
        print()
    
    if all_quarterly_data:
        # 合并季度财务数据
        print("正在合并季度财务数据...")
        quarterly_df = pd.concat(all_quarterly_data, ignore_index=True)
        
        print(f"✅ 季度财务数据合并完成!")
        print(f"   总记录数: {len(quarterly_df)}")
        print(f"   股票数量: {len(quarterly_df['股票代码'].unique())}")
        
        return quarterly_df
    
    return None

def merge_trading_and_financial_data(combined_data, quarterly_data, output_dir="combined_data"):
    """
    合并日度交易数据和季度财务数据
    
    Args:
        combined_data (DataFrame): 合并的交易数据
        quarterly_data (DataFrame): 季度财务数据
        output_dir (str): 输出目录
    """
    
    print("\n" + "=" * 60)
    print("开始合并交易数据和财务数据...")
    print("-" * 60)
    
    if combined_data is None or quarterly_data is None:
        print("❌ 缺少必要的数据，无法进行合并")
        return None
    
    try:
        # 处理季度财务数据，提取关键指标
        print("正在处理季度财务数据...")
        
        # 获取所有可用的财务指标
        all_metrics = quarterly_data['指标'].unique().tolist()
        # 过滤掉'指标'这个列名本身
        key_metrics = [metric for metric in all_metrics if metric != '指标']
        
        print(f"发现 {len(key_metrics)} 个财务指标，将全部进行合并")
        quarterly_processed = []
        
        for stock_code in quarterly_data['股票代码'].unique():
            stock_financial = quarterly_data[quarterly_data['股票代码'] == stock_code]
            
            # 获取最新季度的数据
            latest_quarter = None
            latest_date = None
            
            # 找到最新的季度数据
            for col in stock_financial.columns:
                if col.endswith('1231') or col.endswith('0930') or col.endswith('0630') or col.endswith('0331'):
                    if not stock_financial[col].isna().all():
                        try:
                            date_str = col
                            if date_str > (latest_date or ''):
                                latest_date = date_str
                                latest_quarter = col
                        except:
                            continue
            
            if latest_quarter:
                # 提取最新季度的关键指标
                stock_metrics = {
                    '股票代码': stock_code,
                    '财务数据季度': latest_quarter,
                    '更新时间': stock_financial['更新时间'].iloc[0]
                }
                
                for metric in key_metrics:
                    metric_data = stock_financial[stock_financial['指标'] == metric]
                    if not metric_data.empty and latest_quarter in metric_data.columns:
                        value = metric_data[latest_quarter].iloc[0]
                        if pd.notna(value):
                            stock_metrics[f'{metric}'] = value
                
                quarterly_processed.append(stock_metrics)
        
        # 处理多个季度的数据，为每个季度创建记录
        quarterly_processed_multi = []
        for stock_code in quarterly_data['股票代码'].unique():
            stock_financial = quarterly_data[quarterly_data['股票代码'] == stock_code]
            
            # 获取所有可用的季度
            available_quarters = []
            for col in stock_financial.columns:
                if col.endswith('1231') or col.endswith('0930') or col.endswith('0630') or col.endswith('0331'):
                    if not stock_financial[col].isna().all():
                        available_quarters.append(col)
            
            # 为每个季度创建记录
            for quarter in sorted(available_quarters):
                stock_metrics = {
                    '股票代码': stock_code,
                    '财务数据季度': quarter,
                    '更新时间': stock_financial['更新时间'].iloc[0]
                }
                
                for metric in key_metrics:
                    metric_data = stock_financial[stock_financial['指标'] == metric]
                    if not metric_data.empty and quarter in metric_data.columns:
                        value = metric_data[quarter].iloc[0]
                        if pd.notna(value):
                            stock_metrics[f'{metric}'] = value
                
                quarterly_processed_multi.append(stock_metrics)
        
        quarterly_df = pd.DataFrame(quarterly_processed_multi)
        
        # 为交易数据添加季度标识
        print("正在为交易数据添加季度标识...")
        
        def get_quarter(date):
            """根据日期确定季度"""
            if date.month <= 3:
                return f"{date.year}0331"
            elif date.month <= 6:
                return f"{date.year}0630"
            elif date.month <= 9:
                return f"{date.year}0930"
            else:
                return f"{date.year}1231"
        
        # 重置索引以便操作
        combined_data_reset = combined_data.reset_index()
        combined_data_reset['财务数据季度'] = combined_data_reset['Date'].apply(get_quarter)
        
        # 合并数据
        print("正在合并交易数据和财务数据...")
        merged_data = combined_data_reset.merge(
            quarterly_df, 
            left_on=['Stock_Code', '财务数据季度'], 
            right_on=['股票代码', '财务数据季度'],
            how='left'
        )
        
        # 删除重复的股票代码列
        if '股票代码' in merged_data.columns:
            merged_data = merged_data.drop('股票代码', axis=1)
        
        # 不填充财务数据，保持严格的季度对应关系
        print("财务数据已按季度严格对应，未填充空值")
        
        # 重新设置日期索引
        merged_data = merged_data.set_index('Date')
        
        # 保存合并后的数据
        filename = "merged_trading_financial_data.csv"
        filepath = os.path.join(output_dir, filename)
        merged_data.to_csv(filepath, encoding='utf-8-sig')
        
        print(f"✅ 数据合并完成!")
        print(f"   总记录数: {len(merged_data)}")
        print(f"   股票数量: {len(merged_data['Stock_Code'].unique())}")
        print(f"   包含财务指标: {len([col for col in merged_data.columns if col in key_metrics])}")
        print(f"   保存至: {filepath}")
        
        # 显示数据样本
        print("\n合并数据预览:")
        display_cols = ['Stock_Code', 'Open', 'Close', 'Volume', '财务数据季度'] + [col for col in key_metrics if col in merged_data.columns]
        available_cols = [col for col in display_cols if col in merged_data.columns]
        print(merged_data[available_cols].head(5).to_string())
        
        # 显示财务指标统计
        financial_cols = [col for col in key_metrics if col in merged_data.columns]
        print(f"\n成功合并的财务指标 ({len(financial_cols)} 个):")
        for i, col in enumerate(financial_cols, 1):
            non_null_count = merged_data[col].notna().sum()
            print(f"  {i:2d}. {col} (非空值: {non_null_count})")
        
        return merged_data
        
    except Exception as e:
        print(f"❌ 数据合并失败: {str(e)}")
        return None

def load_stock_codes_from_file(filename):
    """
    从TXT文件读取股票代码列表
    
    Args:
        filename (str): 文件名
        
    Returns:
        list: 股票代码列表
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        stock_codes = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):  # 忽略空行和注释行
                stock_codes.append(line)
        
        return stock_codes
    except FileNotFoundError:
        print(f"❌ 文件 {filename} 不存在")
        return []
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return []

def get_date_range():
    """
    获取日期范围
    
    Returns:
        tuple: (start_date, end_date)
    """
    print("请选择时间范围输入方式:")
    print("1. 手动输入日期范围")
    print("2. 使用默认日期范围")
    print("3. 使用预设时间段")
    print()
    
    while True:
        choice = input("请选择 (1/2/3): ").strip()
        
        if choice == '1':
            print()
            print("手动输入日期范围")
            print("-" * 30)
            print("日期格式: YYYY-MM-DD")
            print("示例: 2024-01-01")
            print()
            
            while True:
                start_date = input("请输入开始日期 (YYYY-MM-DD): ").strip()
                if not start_date:
                    print("❌ 请输入开始日期")
                    continue
                
                # 验证日期格式
                try:
                    datetime.strptime(start_date, '%Y-%m-%d')
                    break
                except ValueError:
                    print("❌ 日期格式错误，请使用 YYYY-MM-DD 格式")
                    continue
            
            while True:
                end_date = input("请输入结束日期 (YYYY-MM-DD): ").strip()
                if not end_date:
                    print("❌ 请输入结束日期")
                    continue
                
                # 验证日期格式
                try:
                    datetime.strptime(end_date, '%Y-%m-%d')
                    break
                except ValueError:
                    print("❌ 日期格式错误，请使用 YYYY-MM-DD 格式")
                    continue
            
            # 验证日期逻辑
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                if start_dt >= end_dt:
                    print("❌ 开始日期必须早于结束日期")
                    continue
                
                # 检查是否超过当前日期
                if end_dt > datetime.now():
                    print("⚠️  结束日期超过当前日期，可能导致数据不完整")
                    confirm = input("是否继续? (y/n): ").strip().lower()
                    if confirm not in ['y', 'yes', '是']:
                        continue
                
                return start_date, end_date
                
            except Exception as e:
                print(f"❌ 日期验证失败: {e}")
                continue
                
        elif choice == '2':
            print()
            print("使用默认日期范围")
            print("-" * 30)
            start_date = '2024-01-01'
            end_date = '2024-12-31'
            print(f"开始日期: {start_date}")
            print(f"结束日期: {end_date}")
            return start_date, end_date
            
        elif choice == '3':
            print()
            print("使用预设时间段")
            print("-" * 30)
            print("1. 最近一年 (2024-01-01 至 2024-12-31)")
            print("2. 最近半年 (2024-07-01 至 2024-12-31)")
            print("3. 最近三个月 (2024-10-01 至 2024-12-31)")
            print("4. 今年至今 (2024-01-01 至 今日)")
            print("5. 去年全年 (2023-01-01 至 2023-12-31)")
            print()
            
            while True:
                preset_choice = input("请选择预设时间段 (1-5): ").strip()
                
                current_date = datetime.now()
                
                if preset_choice == '1':
                    start_date = '2024-01-01'
                    end_date = '2024-12-31'
                elif preset_choice == '2':
                    start_date = '2024-07-01'
                    end_date = '2024-12-31'
                elif preset_choice == '3':
                    start_date = '2024-10-01'
                    end_date = '2024-12-31'
                elif preset_choice == '4':
                    start_date = '2024-01-01'
                    end_date = current_date.strftime('%Y-%m-%d')
                elif preset_choice == '5':
                    start_date = '2023-01-01'
                    end_date = '2023-12-31'
                else:
                    print("❌ 请输入 1-5 之间的数字")
                    continue
                
                print(f"选择的时间段: {start_date} 至 {end_date}")
                return start_date, end_date
                
        else:
            print("❌ 请输入 1、2 或 3")

def get_stock_codes():
    """
    获取股票代码列表
    
    Returns:
        list: 股票代码列表
    """
    print("请选择输入股票代码的方式:")
    print("1. 手动输入股票代码")
    print("2. 从TXT文件读取股票代码列表")
    print()
    
    while True:
        choice = input("请选择 (1/2): ").strip()
        
        if choice == '1':
            print()
            print("手动输入股票代码")
            print("-" * 30)
            print("格式说明:")
            print("- 多个股票代码用逗号分隔")
            print("- 支持A股代码格式: 600030, 002261, 688326, 300364")
            print("- 示例: 600030,002261,688326,300364")
            print()
            
            stock_input = input("请输入股票代码: ").strip()
            if not stock_input:
                print("❌ 请输入有效的股票代码")
                continue
            
            # 解析股票代码
            stock_codes = [code.strip() for code in stock_input.split(',')]
            stock_codes = [code for code in stock_codes if code]  # 移除空字符串
            
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
            
        elif choice == '2':
            print()
            print("从TXT文件读取股票代码")
            print("-" * 30)
            print("文件格式说明:")
            print("- 每行一个股票代码")
            print("- 支持注释行（以#开头）")
            print("- 示例文件内容:")
            print("  # 这是注释行")
            print("  600030")
            print("  002261")
            print("  688326")
            print("  300364")
            print()
            
            filename = input("请输入TXT文件名 (例如: stock_codes.txt): ").strip()
            if not filename:
                print("❌ 请输入文件名")
                continue
            
            # 如果用户没有输入扩展名，自动添加.txt
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            stock_codes = load_stock_codes_from_file(filename)
            if not stock_codes:
                print("❌ 未能从文件中读取到有效的股票代码")
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
            
        else:
            print("❌ 请输入 1 或 2")

def main():
    """主函数"""
    
    print("A股数据合并和财务指标分析脚本")
    print("=" * 60)
    print()
    
    # 获取股票代码列表
    stock_codes = get_stock_codes()
    
    print()
    print("=" * 60)
    print()
    
    # 获取时间范围
    start_date, end_date = get_date_range()
    
    # 输出目录
    output_dir = 'Data_download'
    
    print()
    print("=" * 60)
    print(f"股票代码: {', '.join(stock_codes)}")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 直接开始下载
    print("开始下载和合并数据...")
    
    # 1. 下载并合并交易数据
    combined_data = download_and_merge_stock_data(stock_codes, start_date, end_date, output_dir)
    
    # 2. 下载季度财务数据
    quarterly_data = download_quarterly_financial_data(stock_codes, output_dir)
    
    # 3. 合并交易数据和财务数据
    if quarterly_data is not None:
        merged_data = merge_trading_and_financial_data(combined_data, quarterly_data, output_dir)
    else:
        print("⚠️  季度财务数据下载失败，无法进行数据合并")
        merged_data = None
    
    print("\n" + "=" * 60)
    print("🎉 所有操作完成!")
    print(f"数据保存在: {os.path.abspath(output_dir)}")
    print("\n生成的文件:")
    print("- combined_stock_data.csv: 合并的交易数据")
    if merged_data is not None:
        print("- merged_trading_financial_data.csv: 合并的交易和财务数据（包含70个财务指标）")

if __name__ == "__main__":
    main()
