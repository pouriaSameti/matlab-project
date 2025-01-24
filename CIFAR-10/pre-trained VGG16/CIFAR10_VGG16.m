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
testBatch = load('/MATLAB Drive/cifar-10-batches-mat/test_batch.mat');

if canUseGPU
    disp('GPU is available.');
else
    disp('No GPU detected. Training will proceed on CPU.');
end

train_images = cat(4, ...
    reshape(batch1.data, 32, 32, 3, []), ...
    reshape(batch2.data, 32, 32, 3, []), ...
    reshape(batch3.data, 32, 32, 3, []), ...
    reshape(batch4.data, 32, 32, 3, []), ...
    reshape(batch5.data, 32, 32, 3, []));
train_labels = [batch1.labels; batch2.labels; batch3.labels; batch4.labels; batch5.labels];

test_images = reshape(testBatch.data, 32, 32, 3, []);
test_labels = [testBatch.labels];

disp(['Training images size: ', num2str(size(train_images))]); % Should be [32, 32, 3, 50000];
disp(['Training labels size: ', num2str(size(train_labels))]); % Should be [50000, 1];
disp(['Testing images size: ', num2str(size(test_images))]); % Should be [32, 32, 3, 10000];
disp(['Training labels size: ', num2str(size(test_labels))]); % Should be [10000, 1];

%% Resize images to 224x224 to match VGG16
train_images = imresize(train_images, [224, 224]);
test_images = imresize(test_images, [224, 224]);

train_labels = categorical(train_labels);
test_labels = categorical(test_labels);

%% Data augmentation
augmenter = imageDataAugmenter( ...
    'RandRotation', [-5, 5], ...
    'RandXTranslation', [-2, 2], ...
    'RandYTranslation', [-2, 2]);
augTrainData = augmentedImageDatastore([224 224 3], train_images, train_labels, ...
    'DataAugmentation', augmenter);
augValidationData = augmentedImageDatastore([224 224 3], test_images, test_labels);


%% Load pre-trained VGG16 network
net = vgg16;
lgraph = layerGraph(net);


%% Replace input layer for CIFAR-10
inputLayer = imageInputLayer([224 224 3], 'Name', 'input', 'Normalization', 'zerocenter');
lgraph = replaceLayer(lgraph, 'input', inputLayer);

%% Replace the fully connected layers for CIFAR-10
fc6 = fullyConnectedLayer(4096, 'Name', 'fc6', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
fc7 = fullyConnectedLayer(4096, 'Name', 'fc7', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
fc8 = fullyConnectedLayer(10, 'Name', 'fc8', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
softmaxL = softmaxLayer('Name', 'softmax');
outputLayer = classificationLayer('Name', 'output');

lgraph = replaceLayer(lgraph, 'fc6', fc6);
lgraph = replaceLayer(lgraph, 'fc7', fc7);
lgraph = replaceLayer(lgraph, 'fc8', fc8);
lgraph = replaceLayer(lgraph, 'prob', softmaxL);
lgraph = replaceLayer(lgraph, 'output', outputLayer);

%% Freeze initial layers
layersToFreeze = {'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'};
for i = 1:numel(layersToFreeze)
    layer = lgraph.Layers(find(arrayfun(@(l) strcmp(l.Name, layersToFreeze{i}), lgraph.Layers)));
    if isprop(layer, 'WeightLearnRateFactor')
        layer.WeightLearnRateFactor = 0;
        layer.BiasLearnRateFactor = 0;
        lgraph = replaceLayer(lgraph, layersToFreeze{i}, layer);
    end
end

%% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidationData, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train the network
netTransfer = trainNetwork(augTrainData, lgraph, options);
