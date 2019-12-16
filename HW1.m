load('TrainingSamplesDCT_8.mat');

%A
pri_f=size(TrainsampleDCT_FG,1)...
    /(size(TrainsampleDCT_BG,1)+size(TrainsampleDCT_FG,1));
pri_b=1-pri_f;

%B
%take absolute vlalues
bg=abs(TrainsampleDCT_BG);
fg=abs(TrainsampleDCT_FG);
%find 2nd largest
sortb= sort(bg,2, 'descend');
sortf= sort(fg,2,'descend');
%disp(size(sortb,1)
for i=1:size(bg,1)
    for j=1:size(bg,2)
        if(sortb(i,2)==bg(i,j))
            b_ind(i)=j;
        end
    end
end
for i=1:size(fg,1)
    for j=1:size(fg,2)
        if(sortf(i,2)==fg(i,j))
            f_ind(i)=j;
        end
    end
end
b_hist=histogram(b_ind,1:size(bg,2),'Normalization','pdf');
xlabel({'$X$'},'Interpreter','latex');
ylabel({'$P_{X|Y}(x|grass)$'},'Interpreter','latex');
figure();
f_hist=histogram(f_ind,1:size(fg,2),'Normalization','pdf');
xlabel({'$X$'},'Interpreter','latex');
ylabel({'$P_{X|Y}(x|cheetah)$'},'Interpreter','latex');

%C 
zig=load('Zig-Zag Pattern.txt')+1;
cheetah= im2double(imread('cheetah.bmp'));
%clear cheetah_p
cheetah_p=padarray(cheetah,[4,4],0,'pre');
cheetah_p=padarray(cheetah_p,[3,3],0,'post');
for i=1:size(cheetah_p,1)-7
    for j=1:size(cheetah_p,2)-7
       cheetah_dct=abs(dct2(cheetah_p(i:i+7, j:j+7)));
       for k=1:64
           [z1,z2]=find(zig==k);
           X_dct(k)=cheetah_dct(z1,z2);
       end
       sortX=sort(X_dct, 'Descend');
       for m=1:64 %find feature X
           if(X_dct(m)==sortX(2))
               X(i,j)=m;
           end
       end
    end
end 

%Using MAP
clear b_prob  f_prob
b_prob=histcounts(b_ind,1:65)/size(TrainsampleDCT_BG,1);
f_prob=histcounts(f_ind,1:65)/size(TrainsampleDCT_FG,1);

for i=1:size(cheetah,1)
    for j=1:size(cheetah,2)
        %disp(X(i,j))
        if(b_prob(X(i,j))*pri_b>=f_prob(X(i,j))*pri_f)
            final(i,j)=0;
        else
            final(i,j)=1;
        end
    end
end

figure();
imshow(final)

for i=2:64
    post(1,i)=b_prob(i)*pri_b;
    post(2,i)=f_prob(i)*pri_f;
end
 
%D, calcuate error and Bayes Error (Risk)
truth=imread('cheetah_mask.bmp');
truth=im2double (truth);
err=0;

for i=1:size(truth,1)
    for j=1:size(truth,2)
        if (final(i,j)~= truth(i,j))
            err=err+1;
        end
    end
end

%
bn=1;
fn=1;
 if(truth(i,j)==0)
                b_err(bn)=X(i,j);
                bn=bn+1;
            elseif(b_prob(X(i,j))*pri_b~=0 && f_prob(X(i,j))*pri_f~=0)
                f_err(fn)=X(i,j);
                fn=fn+1;
                end
err_rate=err/(size(truth,1)*size(truth,2));
disp(err_rate)
bay_b_err=0;
bay_f_err=0;
b_list=unique(b_err);
f_list=unique(f_err);
for i=1:length(b_list)
    bay_b_err=bay_b_err+b_prob(1,b_list(i));
end
for i=1:length(f_list)
    bay_f_err=bay_f_err+f_prob(1,f_list(i));
end
risk_MAP=pri_b*bay_b_err+pri_f*bay_f_err
%




