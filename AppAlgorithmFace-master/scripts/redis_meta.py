'''redis'''
import requests
import redis


def get_redis_meta():
    # redis-cli -h 10.69.12.245 -p 9745 -a test
    fat = ["https://fat-basicconf.hellobike.cn/redis/getConfig", "public-redis", "1Ji7WkN14NWRcZ30oND0"]
    # redis-cli -h 10.69.12.209 -p 9558 -a test
    uat = ["https://uat-basicconf.hellobike.cn/redis/getConfig", "public-redis", "nQX85luVPt2KWPWybdQb"]
    # redis-cli -h aiengine-algorithm-bikebattery-redis.ttbike.com.cn -p 8853 -a bLQqu59GNos27HLmq
    pro = ["http://basic-conf-inner.hellobike.cn/redis/getConfig", "aiengine-algorithm-bikebattery", "U2S8WRdQjgWNBZMK6Y1B"]
    url, redis_name, token = fat
    r = requests.post(url,
                      headers={
                          "REDIS_PROJECT_APPID": "AppAlgorithmBikeBattery",
                          "REDIS_PROJECT_TOKEN": token
                      },
                      json={"redisList": [redis_name]})
    assert r.status_code == 200, r.content
    print(r.json())


if __name__ == "__main__":
    get_redis_meta()

