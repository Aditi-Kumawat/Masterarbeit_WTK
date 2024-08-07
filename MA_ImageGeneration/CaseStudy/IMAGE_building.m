% Specify the path to your BMP file
filename = '6.bmp';

% Read the BMP file
imageData = imread(filename);

% Display the image
figure;
imshow(imageData);
hold on
%title('Imported BMP Image');
scatter(450,470,'black','filled')
