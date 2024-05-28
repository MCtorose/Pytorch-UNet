x1=223;
x2=63;
% 初始化x轴坐标值
initialx=360-x1/2;
% 初始化y轴坐标值
initialy=240-x2/2;
path='E:\train_image\jpg';
% 创建图像数据存储对象
imdspath = imageDatastore(path);
% 返回总共的焊缝图片的数量
numTest = length(imdspath.Files);
hold on;
for i=1:numTest
    % subplot(numTest,1,i);
    a=readimage(imdspath,i);
    imshow(a)
    a=imcrop(a,[initialx initialy x1 x2]);
    base=a;
    imshow(a)
    save_path_prefix = 'E:\Desktop\Matlab_DeepL\split_png\';
    result_png_path = strcat(save_path_prefix,['result_' num2str(i) '.png']);
    imwrite(a, result_png_path);
    % 中值滤波的窗口大小为4x4
    a=medfilt2(a,[3 3]);
    imshow(a);
    %阈值分割
    a=a<30;
    % imshow(a)
    %腐蚀
    %创建一个方形（square）形状的结构元素，其大小为 8
    se=strel('square',2);
    a=imerode(a,se);
    % imshow(a)
    %将区域联通
    a = bwlabel(a,8);
    % imshow(a)
    % 提取图像的轮廓
    contour_img = bwperim(a);
    % 获取轮廓的坐标
    [rows, cols] = find(contour_img);
    % 绘制轮廓
    imshow(base);
    hold on;
    plot(cols, rows, 'r.'); % 绘制红色点表示轮廓的坐标
    hold off;
    % 定义 points、imagePath 和 imageData
    points=[cols, rows];
    points(end,:)=[];
    % % 读取图像文件
    % % 将图像数据编码为 Base64 字符串
    [base64string,base64string_len] = base64file(result_png_path);
    disp(base64string);
    % 构建 JSON 结构
    json_data = struct(...
        'version', '5.1.1', ...
        'flags', struct(), ...
        'shapes', [struct('label','hanfeng','points',points,'group_id',[],'shape_type','polygon','flags',struct()),struct('label','hanfeng','points',points,'group_id',[],'shape_type','polygon','flags',struct())], ...
        'imagePath', result_png_path, ...
        'imageData', base64string, ... % 将 Base64 字符串包含在 JSON 内容中
        'imageHeight', size(a, 1), ... % 获取图像高度
        'imageWidth', size(a, 2) ... % 获取图像宽度
        );
    % % 编码为 JSON 格式
    json_str = jsonencode(json_data);
    % 保存 JSON 字符串到文件
    json_save_path_prefix = 'E:\train_image\json\';
    result_png_json_path = strcat(json_save_path_prefix,['result_' num2str(i) '.json']);
    fid = fopen(result_png_json_path, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    disp('JSON 文件保存完成。');
end
hold off;
