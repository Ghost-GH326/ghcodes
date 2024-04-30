import json5

# def replace_single_quotes_with_double_quotes(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json5.load(file)

#     # 将单引号替换为双引号
#     modified_data = replace_quotes_recursive(data)

#     with open(file_path, 'w', encoding='utf-8') as file:
#         json5.dump(modified_data, file, ensure_ascii=False, indent=2)

# def replace_quotes_recursive(obj):
#     if isinstance(obj, dict):
#         return {replace_quotes_recursive(key): replace_quotes_recursive(value) for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [replace_quotes_recursive(element) for element in obj]
#     elif isinstance(obj, str):
#         return obj.replace("'","\"")
#     else:
#         return obj


def replace_single_quotes_with_double_quotes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 将单引号替换为双引号
    modified_content = content.replace('\\"', "")

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

# 替换文件中的所有单引号
file_path = '/home/guohao826/AppAlgorithmFace/models/feature_model/train.json5'  
replace_single_quotes_with_double_quotes(file_path)
