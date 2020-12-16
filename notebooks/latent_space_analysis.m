close all; clear;

%% Read in Data
emb=table2array(readtable('~/Documents/cmufall2020/ml/project/data/test_embed_linear.txt'));

test_x=table2array(readtable('~/Documents/cmufall2020/ml/project/data/test_x.txt'));

labels=table2array(readtable('~/Documents/cmufall2020/ml/project/data/test_labels_linear.txt'));
[~,test_labels]=max(labels,[],2);

bb_coords=table2array(readtable('~/Documents/cmufall2020/ml/project/data/bbcoord_linear_s_bb.txt'));

emotions=["anger", "disgust", "fear", "happiness", "sadness", "surprise","neutral", "contempt", "unknown", "NF"];

%% Second order-interactions
emb_ord2=[];
for dim = 1:size(emb,2)
    tmp=emb.*emb(:,dim);
    emb_ord2=[emb_ord2,tmp];
end

%% Binary Test Labels
test_labels_binary=[];
for class = 1:10
    tmp=test_labels==class;
    test_labels_binary=[test_labels_binary,tmp];
end

corr_ord1=abs(corr(emb,test_labels_binary));
corr_ord2=abs(corr(emb_ord2,test_labels_binary));
max_corr=max([corr_ord1(:);corr_ord2(:)]);

figure;
imagesc(corr_ord1,[0,max_corr])
set(gca,'XTick',1:10)
set(gca,'XTickLabel',emotions)
ylabel('Latent Dimension')
colormap('redbluecmap')
colorbar

figure;
imagesc(corr_ord2,[0,max_corr])
set(gca,'XTick',1:10)
set(gca,'XTickLabel',emotions)
ylabel('2nd Order Latent Dimension')
colormap('redbluecmap')
colorbar

%% Bar graphs
[corr_ord1, p_ord1]=corr(emb,test_labels_binary);
[corr_ord2, p_ord2]=corr(emb_ord2,test_labels_binary);

p_ord1=reshape(mafdr(p_ord1(:)),size(corr_ord1)); p_ord2=reshape(mafdr(p_ord2(:)),size(corr_ord2));

corr_ord1(p_ord1>0.05)=0; max_ord1=max(abs(corr_ord1));
corr_ord2(p_ord2>0.05)=0; max_ord2=max(abs(corr_ord2));
figure; bar([max_ord1',max_ord2'])
set(gca,'XTick',1:10)
set(gca,'XTickLabel',emotions)
legend(["Latent Dimensions","2nd Order Interactions"]);
ylim([0,0.8])

%% PCA
[coeff,score]=pca(emb);

x2=score(:,1);
y2=score(:,2);

figure;
scatter(x2(test_labels==1,:),y2(test_labels==1,:),15,[1,0,0],'filled');
hold on
scatter(x2(test_labels==2,:),y2(test_labels==2,:),15,[0,1,0],'filled');
scatter(x2(test_labels==3,:),y2(test_labels==3,:),15,[0,0,1],'filled');
scatter(x2(test_labels==4,:),y2(test_labels==4,:),15,[1,1,0],'filled');
scatter(x2(test_labels==5,:),y2(test_labels==5,:),15,[0,0,0],'filled');
scatter(x2(test_labels==6,:),y2(test_labels==6,:),15,[0,1,1],'filled');
scatter(x2(test_labels==7,:),y2(test_labels==7,:),15,[1,0,1],'filled');
scatter(x2(test_labels==8,:),y2(test_labels==8,:),15,[0,0.5,0.5],'filled');
scatter(x2(test_labels==9,:),y2(test_labels==9,:),15,[0.5,0,0.5],'filled');
scatter(x2(test_labels==10,:),y2(test_labels==10,:),15,[0.5,0.5,0],'filled');

legend(emotions)
xlabel('Scores on PC1')
ylabel('Scores on PC2')

%% Plot Bounding Box Correlations
figure;
count=1;
bb_names={"Box x-coordinate","Box y-coordinate","Box height","Box length"};
for bb_params=1:4
    subplot(2,4,count)
    scatter(bb_coords(:,bb_params),x2,5,'black','filled')
    xlabel(bb_names{bb_params})
    ylabel('Principal Component 1')
    count=count+1;
end
for bb_params=1:4
    subplot(2,4,count)
    scatter(bb_coords(:,bb_params),y2,5,'black','filled')
    xlabel(bb_names{bb_params})
    ylabel('Principal Component 2')
    count=count+1;
end

%% Eigenface (source: Wikipedia)
h = 48; w = 48;

% vectorize images
x = test_x';
x = double(x);

% subtract mean
mean_matrix = mean(x, 2);
x = bsxfun(@minus, x, mean_matrix);

% calculate covariance
s = cov(x');

% obtain eigenvalue & eigenvector
[V, D] = eig(s);
eigval = diag(D);

% sort eigenvalues in descending order
eigval = eigval(end: - 1:1);
V = fliplr(V);

%% Plot
% show mean and 1st through 9th principal eigenvectors
figure;
for i = 1:9
    subplot(3, 3, i)
    imagesc(reshape(V(:, i), h, w)')
    set(gca,'xticklabel',[]); set(gca,'yticklabel',[])
    title(strcat("Eigenface ",string(i)))
    colormap gray
end

%% Project onto eigenfaces
weights = zeros(size(x,2),25);

for eigface_sel = 1:25
    for idx = 1:size(x,2)
        tmp_w = V(:,eigface_sel)'*(x(:,idx)-mean_matrix);
        weights(idx,eigface_sel) = abs(tmp_w);
    end
end

corr_ord1=abs(corr(emb,weights));
corr_ord2=abs(corr(emb_ord2,weights));
max_corr=max([corr_ord1(:);corr_ord2(:)]);

figure;
imagesc(corr_ord1,[0,max_corr])
xlabel('Eigenface Index')
ylabel('Latent Dimension')
colormap('redbluecmap')
colorbar

figure;
imagesc(corr_ord2,[0,max_corr])
xlabel('Eigenface Index')
ylabel('2nd Order Latent Dimension')
colormap('redbluecmap')
colorbar

%% Bar graphs
[corr_ord1, p_ord1]=corr(emb,weights);
[corr_ord2, p_ord2]=corr(emb_ord2,weights);

p_ord1=reshape(mafdr(p_ord1(:)),size(corr_ord1)); p_ord2=reshape(mafdr(p_ord2(:)),size(corr_ord2));

corr_ord1(p_ord1>0.05)=0; max_ord1=max(abs(corr_ord1));
corr_ord2(p_ord2>0.05)=0; max_ord2=max(abs(corr_ord2));
figure; bar([max_ord1',max_ord2'])
xlabel("Eigenface Index")
ylabel("Pearson Correlation")
legend(["Latent Dimensions","2nd Order Interactions"]);

%% Inspect eigenfaces
count=1;
for eigface_sel = 1:10
    subplot(10,11,count)
    imagesc(reshape(V(:, eigface_sel), h, w)')
    title(strcat("Eigneface ",string(eigface_sel)))
    set(gca,'xticklabel',[]); set(gca,'yticklabel',[])
    count = count + 1;
    
    tmp_idx = find(weights(:,eigface_sel)>prctile(weights(:,eigface_sel),99));
    tmp_idx = tmp_idx(1:10);
    for idx = 1:10
        subplot(10,11,count)
        imagesc(reshape(x(:, tmp_idx(idx)), h, w)')
        set(gca,'xticklabel',[]); set(gca,'yticklabel',[])
        count = count + 1;
    end
end
