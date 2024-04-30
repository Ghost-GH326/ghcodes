'''
图片 service，用于图片处理、上传、下载等
'''
from src.util import http


class ImageService:
    @classmethod
    async def download_image(cls, url):
        async with http.session.get(url) as resp:
            assert resp.status == 200, "download image error: {}".format(
                resp.status)
            return await resp.read()
