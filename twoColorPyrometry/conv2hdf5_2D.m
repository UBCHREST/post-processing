

filename = dir('*.mat')
for i=1:length(filename)
    currFileName = filename(i).name;
    
    file = currFileName(1:end-4);

    matstruct=load(currFileName);
    mat = vertcat(matstruct.intensity);

    %     assume everything starts at 0
    xstart=0;
    ystart=0;
    zstart=0;
    %     assume discretization is uniform, dx=dy=dz, get dy from chamber height (1in)
    dy=0.0254/83;
    dx=dy;
    dz=dy;

    xend=size(mat,1)*dx
    zend=size(mat,3)*dz


    %files to write
    startm=[xstart zstart]
    endm=[xend zend]
    dxm=[dx dz]

    %     currFileName = 'myfile';
    hdfname=append(file,'.h5')
    if isfile(hdfname)==false
        h5create(hdfname,'/main/intensity',[size(mat,1) size(mat,2)])
        h5create(hdfname,'/main/start',[1 2])
        h5create(hdfname,'/main/end',[1 2])
        h5create(hdfname,'/main/discretization',[1 2])
    end

    h5write(hdfname,"/main/intensity",mat)
    h5write(hdfname,"/main/start",startm)
    h5write(hdfname,"/main/end",endm)
    h5write(hdfname,"/main/discretization",dxm)

end

% h5disp("temperature1.h5")