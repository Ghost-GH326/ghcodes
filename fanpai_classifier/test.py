"""
coding:UTF-8
@Software:PyCharm
@Author:ZWY
"""
from run import init
import csv
import ast


def trans_offline_urls(ori_urls):
    if ori_urls in ['(NULL)',None] or isinstance(ori_urls,float):
        return []
    ori_urls = ori_urls.replace("'", "")
    ori_urls = ori_urls.replace("，", ",")
    trans_infos = ast.literal_eval(ori_urls)
    transed_urls = []
    for i in range(len(trans_infos)):
        url = trans_infos[i]['url']
        transed_urls.append({'url':url,'index':i})
    return transed_urls

def main():
    cfg = {'classify_threshold': 0.5}
    model = init(cfg)
    

    # # 假设CSV文件名为data.csv，其中包含一个名为collect_images的字段，该字段包含了多个URL
    # csv_file = '/Users/hb28100/Downloads/713592cc-2137-43c8-a492-31d4b9d7e6a2-1.csv'

    # # 创建一个空列表来存储从CSV文件中读取的URL
    # urls_list = []

    # # 打开CSV文件进行读取
    # i = 0
    # with open(csv_file, 'r', newline='', encoding='gbk') as file:
    #     csv_reader = csv.DictReader(file)
    #     for idx, row in enumerate(csv_reader):
    #         # 假设CSV文件中的URL存储在名为collect_images的字段中
    #         # 根据你的CSV文件结构进行调整
    #         ago_image_urls = trans_offline_urls(row['collect_images'])
    #         # 假设URL在CSV文件中使用逗号分隔
    #         #ago_image_urls = ago_image_urls.split(',')
    #         for url in ago_image_urls:
    #             i += 1
    #             # 创建一个字典，包含index和url字段，并添加到urls_list列表中
    #             urls_list.append({"index": i, "url": url['url'].strip()})

    # # 更新params字典中的urls字段
    # params["urls"] = urls_list

    # 输出更新后的params字典
    # print(params)

    params = {
  "urls": [
    {
      "index": 0,
      "url": "https://revolution-video.oss-cn-hangzhou.aliyuncs.com/feedback/20240417/d15189a3d42661c514c9871124b5d265.jpg"
    }
  ],
        # "urls": [
        #     {
        #         "index": 0,
        #         "url": "http://revolution-video.cn-hangzhou.oss.aliyuncs.com/feedback/20240326/ee40ca2154f192b36a6e3a13d0609408.jpg?Expires=1714027886&OSSAccessKeyId=LTAI4GCALcdd4VGKhpZrA8yc&Signature=Yvo01oFpsuUfYOCTPBQA2OpubSw%3D"
        #     }
        # ],
        "guid": "3889998025756598533"
        }
    # params = {
    # "urls": [
    #     {
    #         "index": 1,
    #         "url": "https://easybike-image.oss-cn-hangzhou.aliyuncs.com/test/016f60cb52d54d5fa4bda94dc4d25752.jpg?OSSAccessKeyId=LTAIwDP3dFcdWEUd&Expires=1687688427&Signature=GcLvTYP%2FyvwzKzgnADQbcxx32yI%3D"
    #     },
    #     {
    #         "index": 0,
    #         "url": "https://easybike-image.oss-cn-hangzhou.aliyuncs.com/test/9510030592_bos_1_-2.jpg?OSSAccessKeyId=LTAIwDP3dFcdWEUd&Expires=1675593891&Signature=xXHnrwllwCSaO%2FjKELZT%2FNj58VU%3D"
    #     }
    # ],
    # "guid": "3889998025756598533"
    # }

    res = model.predict(params)
    print(res)


if __name__ == '__main__':
    main()