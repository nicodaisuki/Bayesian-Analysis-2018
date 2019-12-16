load('TrainingSamplesDCT_8.mat');
zig=load('Zig-Zag Pattern.txt')+1;
truth=imread('cheetah_mask.bmp');
truth=im2double (truth);
pri_f=size(TrainsampleDCT_FG,1)...
    /(size(TrainsampleDCT_BG,1)+size(TrainsampleDCT_FG,1));
pri_b=1-pri_f;
cheetah= im2double(imread('cheetah.bmp'));
cheetah_p=padarray(cheetah,[4,3],0,'pre');
cheetah_p=padarray(cheetah_p,[3,4],0,'post');
n=1;
cheetah_dct=zeros(68850,64);

for i=1:size(cheetah_p,1)-7
    for j=1:size(cheetah_p,2)-7
        temp=dct2(cheetah_p(i:i+7, j:j+7));
        for k=1:8
            for m=1:8
                cheetah_dct(n,zig(k,m))=temp(k,m);
            end
        end
        n=n+1;
    end
end

dims=[1 2 4 8 16 24 32 40 48 56 64];
n_comp=8; %components
n_mix=5;

mean_bg=zeros(n_mix, n_comp,64);
mean_fg=zeros(n_mix, n_comp,64);
cov_bg=zeros(n_mix, n_comp,64,64);
cov_fg=zeros(n_mix, n_comp,64,64);
pi_bg=zeros(n_mix, n_comp); %weight, pi_c
pi_fg=zeros(n_mix, n_comp);

for i=1:n_mix
    [mean_bg(i,:,:),cov_bg(i,:,:,:),pi_bg(i,:)]=...
        take_em(TrainsampleDCT_BG,n_comp,64,50);
    [mean_fg(i,:,:),cov_fg(i,:,:,:),pi_fg(i,:)]=...
        take_em(TrainsampleDCT_FG,n_comp,64,50);
end
im_set=zeros(n_mix,n_mix,size(dims,2),...
    size(cheetah,1),size(cheetah,2));
err_set=zeros(n_mix,n_mix,size(dims,2),1);
for b=1:n_mix
    for f=1:n_mix
        for d=1:size(dims,2)
            [im_set(b,f,d,:,:), err_set(b,f,d)]=...
                take_im(cheetah_dct,...
                cheetah,pri_b,pri_f,truth,dims(d),...
                mean_bg(b,:,:),cov_bg(b,:,:,:),pi_bg(b,:),...
                mean_fg(f,:,:),cov_fg(f,:,:,:),...
                pi_fg(f,:),n_comp,1);  
            %disp("Error:"+err_set(b,f,d,1))
        end
      %  disp(" ")
    end
end
% a=zeros(size(im_set,4),size(im_set,5));
% for t=1:size(im_set,4)
%     for s=1:size(im_set,5)
%         a(t,s)=im_set(1,1,5,t,s);
%     end
% end
% imshow(a)
figure();
for i=1:n_mix
    for j=1:n_mix
        err=zeros(size(dims,2),1);
        for k=1:size(dims,2)
            err(k)=err_set(i,j,k);
        end
        hold on;
        plot(dims,err)
 
    end
    xlabel('Feature');
    ylabel({'$Error$'},'Interpreter','latex');
    title(['bg#' int2str(i)])
    legend('fg#1','fg#2','fg#3','fg#4','fg#5','location','best')
    hold off;
    figure();
end

n_comp=[1 2 4 8 16];
mean_bg2=cell(1,size(n_comp,2));
cov_bg2=cell(1,size(n_comp,2));
pi_bg2=cell(1,size(n_comp,2));
mean_fg2=cell(1,size(n_comp,2));
cov_fg2=cell(1,size(n_comp,2));
pi_fg2=cell(1,size(n_comp,2));
for i=1:size(n_comp,2)
    [mean_bg2{i},cov_bg2{i},pi_bg2{i}]=...
        take_em(TrainsampleDCT_BG,n_comp(i),64,50);
    [mean_fg2{i},cov_fg2{i},pi_fg2{i}]=...
        take_em(TrainsampleDCT_FG,n_comp(i),64,50);
end

im_set2=cell(size(dims,2));

for i=1:size(n_comp,2)
    for d=1:size(dims,2)
        [im_set2{d}, err_set2(i,d)]=...
            take_im(cheetah_dct,...
            cheetah,pri_b,pri_f,truth,dims(d),...
            mean_bg2{i},cov_bg2{i},pi_bg2{i},...
            mean_fg2{i},cov_fg2{i},...
            pi_fg2{i},n_comp(i),2);
    end
end

for i =1:size(n_comp,2)
    plot(dims,err_set2(i,:))
    
    hold on; 
    
end
legend('1','2','4','8','16','location','best')
disp("end")

function [mu,sig,pi]=take_em(dct,n_c,dim,iter)

    %initialization
    mu_t=-2.5+5*rand(n_c,dim);
    cov_t=cell(1,n_c);
    for k=1:n_c
        cov_t{k}=diag(1+3*rand(dim,1));
    end
    pi=ones(1,n_c);
    pi=pi/sum(pi);
    %EM
    for i=1:iter
        %E

        for k=1:size(dct,1)
            t_x=zeros(1,n_c);
            for j=1:n_c
                t_x(j)=mvnpdf(dct(k,:),mu_t(j,:),cov_t{j})*pi(j);
            end
            t(k,:)=t_x/sum(t_x);
        end

        pi=sum(t)/size(dct,1);

        %Maximize
        for k = 1: n_c
            temp = (dct - mu_t(k,:)).* (t(:, k));
            cov_t{k}=diag(diag(temp' * (dct-mu_t(k,:)) /sum(t(:, k)))+0.00001);
            mu_t(k,:) = sum(dct.* t(:, k))/sum(t(:, k));
        end
    end
    mu=mu_t;
    sig=zeros(k,64,64);
    for k=1:n_c
        sig(k,:,:)=cov_t{k};
    end
end

function [image, err]=take_im(cheetah_dct,...
    cheetah,pri_b,pri_f,truth,dim,...
    mean_bg,cov_bg,pi_bg,mean_fg,cov_fg,pi_fg,n_c,p)
    
    like_b=0;
    like_f=0;
    for i=1:n_c %components
        c_bg=zeros(dim,dim);
        c_fg=zeros(dim,dim);
        mn_bg=zeros(1,dim);
        mn_fg=zeros(1,dim);
        if(p==1)
            for j=1:dim
                for k=1:dim
                    c_bg(j,k)=cov_bg(1,i,j,k);
                    c_fg(j,k)=cov_fg(1,i,j,k);
                end
            end
            for j=1:dim
                mn_bg(1,j)=mean_bg(1,i,j);
                mn_fg(1,j)=mean_fg(1,i,j);
            end
        else
            for j=1:dim
                for k=1:dim
                    c_bg(j,k)=cov_bg(i,j,k);
                    c_fg(j,k)=cov_fg(i,j,k);
                end
            end
            for j=1:dim
                mn_bg(1,j)=mean_bg(i,j);
                mn_fg(1,j)=mean_fg(i,j);
            end
        end

        like_b=like_b+pi_bg(i)*mvnpdf(cheetah_dct(:,1:dim),...
            mn_bg,c_bg);
        like_f=like_f+pi_fg(i)*mvnpdf(cheetah_dct(:,1:dim),...
            mn_fg,c_fg);
    end

    image=zeros(size(cheetah,1),size(cheetah,2));
    %disp(like_b(1:10))
    n=1;
    for i=1:size(cheetah,1)
        for j=1:size(cheetah,2)
            if(like_b(n)*pri_b>=like_f(n)*pri_f)
                image(i,j)=0;
            else
                image(i,j)=1;
            end
            n=n+1;
        end
    end
    %     figure();
    %     imshow(image)
    %     title("ML Set ")

    %calculate error
    err=error(image,truth);
end

function err=error(img,truth)
    %calculate error
    err=0;
    for i=1:size(truth,1)
        for j=1:size(truth,2)
            if (img(i,j)~= truth(i,j))
                err=err+1;
            end
        end
    end
    err=err/(size(truth,1)*size(truth,2));
end

