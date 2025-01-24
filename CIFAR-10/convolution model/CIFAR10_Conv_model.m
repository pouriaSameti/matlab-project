%% Download the CIFAR-10 dataset
if ~exist('cifar-10-batches-mat','dir')
    cifar10Dataset = 'cifar-10-matlab';
    disp('Downloading 174MB CIFAR-10 dataset...');   
    websave([cifar10Dataset,'.tar.gz'],...
        ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
    gunzip([cifar10Dataset,'.tar.gz'])
    delete([cifar10Dataset,'.tar.gz'])
    untar([cifar10Dataset,'.tar'])
    delete([cifar10Dataset,'.tar'])
end  

batch1 = load('/MATLAB Drive/cifar-10-batches-mat/data_batch_1.mat');  % Replace with your uploaded file name
batch2 = load('/MATLAB Drive/cifar-10-batches-mat/data_batch_2.mat');  % Replace with your uploaded file name
batch3 = load('/MATLAB Drive/cifar-10-batches-mat/data_batch_3.mat');  % Replace with your uploaded file name
batch4 = load('/MATLAB Drive/cifar-10-batches-mat/data_batch_4.mat');  % Replace with your uploaded file name
batch5 = load('/MATLAB Drive/cifar-10-batches-mat/data_batch_5.mat');  % Replace with your uploaded file name

if canUseGPU
    disp('GPU is available.');
else
    disp('No GPU detected. Training will proceed on CPU.');
end
data = [batch1.data; batch2.data; batch3.data; batch4.data; batch5.data];
labels = [batch1.labels; batch2.labels; batch3.labels; batch4.labels; batch5.labels];


numImages = size(data, 1);
XTrain = reshape(data, [32, 32, 3, numImages]);
XTrain = permute(XTrain, [2, 1, 3, 4]); 

imageSize = [32 32 3];
XTrain = double(XTrain) / 255;
YTrain = categorical(labels);

testBatch = load('/MATLAB Drive/cifar-10-batches-mat/test_batch.mat');
XTest = reshape(testBatch.data, [32, 32, 3, size(testBatch.data, 1)]);
XTest = permute(XTest, [2, 1, 3, 4]); 
XTest = double(XTest) / 255; 
YTest = categorical(testBatch.labels);


layers = [
    imageInputLayer(imageSize, 'Normalization', 'none', 'Name', 'input')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'dropout1')

    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.5, 'Name', 'dropout2')
    
    fullyConnectedLayer(10, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Specify training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 64, ...
    'MiniBatchSize', 1024, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationData', {XTest, YTest});

% Train the network
net = trainNetwork(XTrain, YTrain, layers, options);