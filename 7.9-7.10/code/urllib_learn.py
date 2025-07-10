#使用urllib获取百度首页的源码
#import urllib.request
#定义一个url 就是你要访问的地址
#url="http://www.baidu.com"
#模拟浏览器向服务器发送请求
#response=urllib.request.urlopen(url)

#获取响应界面中的源码 content内容的意思
#read()方法返回的是字节形式的二进制数据
#我们要将二进制的数据转换成字符串（解码）
#content=response.read().decode("utf-8")
#print(content)

#response类型是HTTPResponse的类型
#print(type(response))

#一个一个字节去读
#content=response.read()

#读五个字节
#content=response.read(5)
#print(content)

#读取一行
#content=response.readline()
#print(content)

#按行全部读
#content=response.readlines()
#print(content)

#返回状态码，200就说明你写的没有问题
#print(response.getcode())

#返回url
#print(response.geturl())

#返回状态信息
#print(response.getheaders())

#下载网页
#url_page="https://www.baidu.com"
#urllib.request.urlretrieve(url_page,"baidu.html")
#下载图片
#url_img="https://img2.baidu.com/it/u=90581149,2158349434&fm=253&fmt=auto&app=138&f=JPEG?w=800&h=1200"
#urllib.request.urlretrieve(url_img,"baidu.jpg")
#下载视频
#url_video=src="https://vdept3.bdstatic.com/mda-qfvaqpnzzj3qf31w/cae_h264/1719738331410781347/mda-qfvaqpnzzj3qf31w.mp4?v_from_s=hkapp-haokan-hbf&auth_key=1752142809-0-0-223414ba9c48b21917eb289934e34545&bcevod_channel=searchbox_feed&pd=1&cr=0&cd=0&pt=3&logid=1209490107&vid=16648779981624158842&klogid=1209490107&abtest="
#urllib.request.urlretrieve(url_video,"bilibili.mp4")

#import urllib.request
#url="https://www.baidu.com"

#headers={
#    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
#}
#因为urlopen方法中不能存储字典，所以需要将字典转换成字符串
#请求对象的定制
#注意因为参数顺序的问题，不能直接写url和headers
#request=urllib.request.Request(url=url,headers=headers)
#response=urllib.request.urlopen(request)
#content=response.read().decode("utf-8")
#print(content)

#需求：获取https://www.baidu.com/s?wd=周杰伦的网页源码

#import urllib.request
#import urllib.parse

#base_url="https://www.baidu.com/s?"
#headers={
#"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
#}

#data={
#    "wd":"周杰伦",
#    "sex":"男",
#    "location":"中国台湾省",
#}

#new_data=urllib.parse.urlencode(data)
#url=base_url+new_data
#print(url)

#request=urllib.request.Request(url=url,headers=headers)
#response=urllib.request.urlopen(request)
#content=response.read().decode("utf-8")
#print(content)

#import json
#import urllib.request
#import urllib.parse

#url="https://fanyi.baidu.com/sug"
#headers={
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
#}
#data={
#    "kw":'spider'
#}
#data=urllib.parse.urlencode(data).encode("utf-8")
#request=urllib.request.Request(url=url,headers=headers,data=data)
#response=urllib.request.urlopen(request)
#content=response.read().decode("utf-8")
#print(content)
#obj=json.loads(content)
#print(obj)













