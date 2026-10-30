#!/usr/bin/env python
import glob
import os,sys
import re

def analyze_clashscore_files(folder_path, file_extension, check_list, t=0):
    # 初始化列表
    list_A = []  # 无CLASHSCORE字段的文件
    list_B = []  # 有CLASHSCORE字段的文件
    list_C = []  # CLASHSCORE >= 10的文件
    list_D = []  # CLASHSCORE < 10的文件
    list_E = []  # 在check_list中但文件夹中不存在的文件
    
    # 获取指定后缀的所有文件
    file_pattern = os.path.join(folder_path, f"*/*{file_extension}")
    files0 = glob.glob(file_pattern)
    files = [] 
    for file_path in files0:
        filename = os.path.basename(file_path)
        prefix = filename.split('_')[0]
        if prefix in check_list:
            files.append(file_path)

    
    # 获取所有实际文件名的前缀（_前的部分）
    existing_prefixes = set()
    for file_path in files:
        filename = os.path.basename(file_path)
        prefix = filename.split('_')[0]
        existing_prefixes.add(prefix)
    
    # 找出check_list中但不在文件夹中的前缀
    list_E = [prefix for prefix in check_list if prefix not in existing_prefixes]
    # breakpoint()
    # 编译正则表达式以匹配CLASHSCORE和后面的数值
    if t==1:
        pattern = r"ALL-ATOM CLASHSCORE\s*:\s*(\d+\.?\d*)"
    else:
        pattern = r"All-atom Clashscore\s*:\s*(\d+\.?\d*)"
    
    for file_path in files:
        filename = os.path.basename(file_path).split('_')[0]
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # 查找CLASHSCORE
            match = re.search(pattern, content)
            
            if match:
                # 找到CLASHSCORE
                clashscore = float(match.group(1))
                list_B.append(filename)
                
                # 根据数值分类
                if clashscore >= 10:
                    list_C.append(filename)
                else:
                    list_D.append(filename)
            else:
                # 未找到CLASHSCORE
                list_A.append(filename)
                
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            
    return list_A, list_B, list_C, list_D, list_E, files

def print_results(list_A, list_B, list_C, list_D, list_E, files):
    print("\nFiles with CLASHSCORE >= 10:")
    for file in list_C:
        print(f"  {file}")
    
    print("\nMissing files (in check_list but not in folder):")
    for prefix in list_E:
        print(f"  {prefix}")
        
    print("\nSummary:")
    print(f"Total files : {len(list_A)}")
    print(f"Total files without CLASHSCORE: {len(files)}")
    print(f"Total files with CLASHSCORE: {len(list_B)}")
    print(f"Total files with CLASHSCORE >= 10: {len(list_C)}")
    print(f"Total files with CLASHSCORE < 10: {len(list_D)}")
    print(f"Total missing files: {len(list_E)}")

def save_results(output_dir, list_A, list_B, list_C, list_D, list_E):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存各个列表到文件
    with open(os.path.join(output_dir, 'no_clashscore.txt'), 'w') as f:
        f.write('\n'.join(list_A))
        
    with open(os.path.join(output_dir, 'with_clashscore.txt'), 'w') as f:
        f.write('\n'.join(list_B))
        
    with open(os.path.join(output_dir, 'clashscore_ge_10.txt'), 'w') as f:
        f.write('\n'.join(list_C))
        
    with open(os.path.join(output_dir, 'clashscore_lt_10.txt'), 'w') as f:
        f.write('\n'.join(list_D))
        
    with open(os.path.join(output_dir, 'missing_files.txt'), 'w') as f:
        f.write('\n'.join(list_E))

def read_check_list_from_file(file_path):
    """
    从txt文件中读取检查列表
    文件格式：每行一个前缀
    返回：前缀列表
    """
    try:
        with open(file_path, 'r') as f:
            # 读取所有行，去除空白字符，并过滤掉空行
            check_list = [line.strip() for line in f if line.strip()]
        return check_list
    except Exception as e:
        print(f"Error reading check list file: {e}")
        return []

def main():
    # 配置参数
    date_name="1031"
    date_name="1117"
    date_name="1119"
    na=sys.argv[1]
    file_extension = f"_{na}.vc"
    file_extension = f"_{na}.pdb"
    # date_name="1120"
    # folder_path = f"/share01/hpc/suqiang/ZhangXF/.af/data/models/jobs_{date_name}/"  # 替换为实际文件夹路径
    # if date_name=="1":
    #     file_extension = "_cf_rf.vc"  # 替换为实际文件后缀
    # else:
    #     file_extension = "_cf.vc"
    # file_extension = "_cf_rf.vc"
    folder_path = "/share01/hpc/zhangxiaoxing/data/jobs_pab"
    folder_path = "/data/jobs/f1/jobs_pab"
    output_dir = folder_path  # 输出目录
    
    check_list_file = f"{folder_path}/all.list"  # 包含需要检查的前缀的文件
    check_list_file = f"{folder_path}/te1.list"  # 包含需要检查的前缀的文件
    # check_list_file = f"{folder_path}/cfdb_id.tsv"  # 包含需要检查的前缀的文件
    
    
    # 从文件读取检查列表
    check_list = read_check_list_from_file(check_list_file)
    
    # 分析文件
    list_A, list_B, list_C, list_D, list_E, files = analyze_clashscore_files(folder_path, file_extension, check_list)
    
    # 打印结果
    print_results(list_A, list_B, list_C, list_D, list_E, files)
    
    # 保存结果
    save_results(output_dir, list_A, list_B, list_C, list_D, list_E)

if __name__ == "__main__":
    main()
