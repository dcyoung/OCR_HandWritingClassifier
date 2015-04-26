function displayFeatures(A, name)

warning off all

% rescale
A = A - mean(A(:));

colormap(gray);

% compute rows, cols
[L, M]=size(A);
sz=sqrt(L);
buf=1;
if floor(sqrt(M))^2 ~= M
    n=ceil(sqrt(M));
    while mod(M, n)~=0 && n<1.2*sqrt(M), n=n+1; end
    m=ceil(M/n);
else
    n=sqrt(M);
    m=n;
end

array=-ones(buf+m*(sz+buf),buf+n*(sz+buf));

k=1;
for i=1:m
    for j=1:n
        if k>M, 
            continue; 
        end
        array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/max(abs(A(:)));
        k=k+1;
    end
end

imagesc(array,'EraseMode','none',[-1 1]);
axis image off

drawnow;

print('-djpeg', strcat(name,'.jpg'));

warning on all
