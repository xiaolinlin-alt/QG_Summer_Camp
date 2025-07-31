import cv2

img1=cv2.imread("1.jpg")
img2=cv2.imread("2.jpg")

#去噪和灰度化处理
img1=cv2.GaussianBlur(img1,(5,5),0)
img2=cv2.GaussianBlur(img2,(5,5),0)

img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#使用SIFT算法提取图像中的关键点和特征描述符
sift=cv2.SIFT.create()
keypoints1,discriptors1=sift.detectAndCompute(img1,None)
keypoints2,discriptors2=sift.detectAndCompute(img2,None)
#使用BFMatcher进行特征匹配
bf=cv2.BFMatcher()
matches=bf.knnMatch(discriptors1,discriptors2,k=2)
#筛选出匹配结果
good=[]
for m,n in matches:
    if m.distance<0.75*n.distance:
        good.append([m])
#绘制匹配结果
img3=cv2.drawMatchesKnn(img1,keypoints1,img2,keypoints2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("result",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
#保存匹配结果
cv2.imwrite("result.jpg",img3)
#计算匹配结果
print("匹配结果数量：",len(good))
#计算匹配结果准确率
print("匹配结果准确率：",len(good)/len(matches))