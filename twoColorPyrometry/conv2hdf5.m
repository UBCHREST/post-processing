

filename = dir('*.mat')
for i=1:length(filename)
    currFileName = filename(i).name;
    file = currFileName(1:end-4);

    matstruct=load(currFileName);

    % march over each field
    fields = fieldnames(matstruct);


    for k=1:numel(fields)
        if( isnumeric(matstruct.(fields{k})) )
            mat = vertcat(matstruct.(fields{k}));


            %     assume everything starts at 0
            xstart=0.01880658333;
            ystart=-0.01164166666582;
            zstart=-5E-3;
            %     assume discretization is uniform, dx=dy=dz, get dy from chamber height (1in)
            delta=0.0762/size(mat,2);
            dx=delta;
            dy=delta;
            dz=delta*2;

            % Flip x
            mat = flip(mat, 1);

            xend=size(mat,1)*dx
            yend=size(mat,2)*dy
            zend=size(mat,3)*dz


            %files to write
            startm=[xstart ystart zstart]
            endm=[xend yend zend]
            dxm=[dx dy dz]

            %     currFileName = 'myfile';
            hdfname=append(file,'.h5')
            hdf5FieldName = append('/main/', fields{k});
            if isfile(hdfname)==false
                h5create(hdfname,hdf5FieldName,[size(mat,1) size(mat,2), size(mat,3)])
                h5create(hdfname,'/main/start',[1 3])
                h5create(hdfname,'/main/end',[1 3])
                h5create(hdfname,'/main/discretization',[1 3])
            end

            h5write(hdfname,hdf5FieldName,mat)
            h5write(hdfname,"/main/start",startm)
            h5write(hdfname,"/main/end",endm)
            h5write(hdfname,"/main/discretization",dxm)
        end
    end
end
