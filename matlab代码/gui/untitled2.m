a=imread(replace("E:\train_image\VOC\JPEGImages\result_1.jpg","JPEGImages\result_1.jpg","result2.png"));
% base=a;
% a=medfilt2(a,[3 3]);
% a=a<30;
% se=strel('square',2);
% a=imerode(a,se);
% a = bwlabel(a,8);
% % 提取图像的轮廓
% contour_img = bwperim(a);
% % 获取轮廓的坐标
% [rows, cols] = find(contour_img);
% % 绘制轮廓
% imshow(base);
% hold on;
% plot(cols, rows, 'r.'); % 绘制红色点表示轮廓的坐标
% hold off;
imshow(a);
title('传统方法')












