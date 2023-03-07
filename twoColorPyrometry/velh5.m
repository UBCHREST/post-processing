clear all;clc;


filename = dir('*.txt');
dx=0.25;
t0=-200000;
dt=10000;
sample_width=7.96;
for i=1:length(filename)

    currFileName = filename(i).name;
    pat = digitsPattern;
    framestring= extract(currFileName,pat);
    t=(str2double( framestring )-t0)/dt;
    M = readmatrix(currFileName);

    %getting number of rows
    [Cx,iarow,icrow] = unique(M(:,1));
    a_counts = accumarray(icrow,1);
    value_counts = [Cx, a_counts];
    numrow=value_counts(1,2);
    % getting number of columns
    [Cy,iacol,iccol] = unique(M(:,2));
    a_counts = accumarray(iccol,1);
    value_counts = [Cy, a_counts];
    numcol=value_counts(1,2);
    
    uvel = zeros(numcol,numrow);
    uvel(sub2ind(size(uvel),icrow,iccol)) = M(:,3);
    
%     uvel=fliplr(flipud(uvel'));
        %uvelplot=uvel./(max(max(uvel)))
        %imshow(uvelplot)


    vvel = zeros(numcol,numrow);
    vvel(sub2ind(size(vvel),icrow,iccol)) = M(:,4);

%     vvel=fliplr(flipud(vvel'));

    xvect=0:dx:100-dx;
    yvect=0:dx:25.5-dx;
    zvect=0:dx:25.5-dx;
    qstuff=1:length(zvect);
    zcent=12.75;
    nstuff=1:length(yvect);
    clear ("M")

    uinterp=zeros(length(xvect),length(yvect),length(zvect));
    vinterp=zeros(length(xvect),length(yvect),length(zvect));
    parfor i=1:length(xvect)
        for n= nstuff
            uinterpval=interp2(Cx.*1000,Cy.*1000,uvel',xvect(i),yvect(n));
            vinterpval=interp2(Cx.*1000,Cy.*1000,vvel',xvect(i),yvect(n));
            for q=qstuff
                if (zvect(q)>(zcent-sample_width/2) && zvect(q)<(zcent+sample_width/2))
                    uinterp(i,n,q)=uinterpval;
                    vinterp(i,n,q)=vinterpval;
                end
            end
        end
    end
    xstart=0;
    ystart=0;
    zstart=-0.01275;
    %     assume discretization is uniform, dx=dy=dz, get dy from chamber height (1in)

    xend=0.1;
    yend=0.0255;
    zend=0.01275;

    %files to write
    startm=[xstart ystart zstart];
    endm=[xend yend zend];
    dxm=[dx dx dx]./1000;
    file ="vel"+sprintf('%.5f',t)
    file=strrep(file,'.','_');
    hdfname=append(file,'.h5')

    usingle=single(uinterp);
    vsingle=single(vinterp);
    veltot(1,:,:,:)=usingle;
    veltot(2,:,:,:)=vsingle;
    if isfile(hdfname)==false
%         h5create(hdfname,'/data/fields/u',size(usingle),'Datatype','single','ChunkSize',size(uinterp),'Deflate',9)
%         h5create(hdfname,'/data/fields/v',size(vsingle),'Datatype','single','ChunkSize',size(uinterp),'Deflate',9)
        h5create(hdfname,'/data/fields/vel',size(veltot),'Datatype','single','ChunkSize',size(veltot),'Deflate',9)
        h5create(hdfname,'/data/grid/start',[1 3])
        h5create(hdfname,'/data/grid/end',[1 3])
        h5create(hdfname,'/data/grid/discretization',[1 3])
    end

%     h5write(hdfname,"/data/fields/u",usingle)
%     h5write(hdfname,"/data/fields/v",vsingle)
    h5write(hdfname,"/data/fields/vel",veltot)
    h5write(hdfname,"/data/grid/start",startm)
    h5write(hdfname,"/data/grid/end",endm)
    h5write(hdfname,"/data/grid/discretization",dxm)
    h5writeatt(hdfname,'/data/','time', t);
    h5writeatt(hdfname,'/data/','oxidizer', 'lowflux cal');


end
