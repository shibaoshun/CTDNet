%% determine the thresholds for segmentation  
imTHZ = imread('dataset\THZ_3.png');
thresh_img = graythresh (imTHZ)*255;  

%% background processing  
pixTrans = 5;          
background_bw = imTHZ > thresh_img;  
Weight = bwdist(1-background_bw);  
Weight(Weight>pixTrans) = pixTrans;  
Weight = Weight/pixTrans;

background_bw = double(background_bw);  
Weight = double(Weight);  
imTHZ = double(imTHZ);  

%% get the image
Mean = sum(sum(Weight.*imTHZ))/sum(sum(Weight));
imBAR = Mean*Weight + imTHZ.*(1 - Weight);
imwrite(imBAR/255,'results\BAR_3.png');



