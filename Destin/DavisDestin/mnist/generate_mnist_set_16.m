nR = 16;
nMnist = 28;


points = [0  0
          0  2
          0  4
          0  6
          0  8
          0  10
          0  12
          2  10
          4  8
          6  6
          8  4
          10 2
          12 0
          12 2
          12 4
          12 6
          12 8
          12 10
          12 12
         ];

points = points + 1;

mnistDataFiles = {'data/train-images-idx3-ubyte', 'data/train-images-idx3-ubyte', 'data/t10k-images-idx3-ubyte'};
mnistLabelFiles = {'data/train-labels-idx1-ubyte', 'data/train-labels-idx1-ubyte', 'data/t10k-labels-idx1-ubyte'};
outDataFiles = {'data/train.bin', 'data/destin_train_nn_16.bin', 'data/destin_test_nn_16.bin'};
outLabelFiles = {'data/labels.bin', 'data/labels_train_nn_16.bin', 'data/labels_test_nn_16.bin'};
nSamples = [60000 60000 10000];

digitsList = 0:2;

for f=1:1
	mnistSet = loadMNISTImages(mnistDataFiles{f})';
    mnistLabels = loadMNISTLabels(mnistLabelFiles{f});
	
	fidData = fopen(outDataFiles{f}, 'w');
    fidLabels = fopen(outLabelFiles{f}, 'w');
	
	for d=1:nSamples(f)
        if sum(mnistLabels(d) == digitsList) == 0
            continue;
        end
            
	    tmpImage = reshape(mnistSet(d,:), nMnist, nMnist);
	    
	    %for i=1:length(points)
        for i=1:3
	        %tmpCropImage = tmpImage(points(i,1):points(i,1)+15, points(i,2):points(i,2)+15)';
            tmpCropImage = imresize(tmpImage, [nR nR])';
            tmpCropImage(tmpCropImage < 0) = 0;
            fwrite(fidLabels, mnistLabels(d), 'uint32');
	        fwrite(fidData, reshape(tmpCropImage, 1, nR*nR), 'float');
        end
	end
	
	fclose(fidData);
end
