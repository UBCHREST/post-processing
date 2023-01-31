% this support script is desinged to convert .m raw files from experiments
% to the chrest hdf5 file format.  

% update the file metadata 
metadata=dictionary();
metadata('fuel')="pmma";
metadata('oxidizer')="O2";
metadata('caseLabel')="20.55G";
metadata('date')='2023-01-23';
metadata('contact')="elektrak@buffalo.edu,kolosret@buffalo.edu";

% specify the time history data
timeStart=0.0;
snapShotDt=1.0/1000;

% update the grid information.  Specify the lower left hand corner of the
% data. The start/end points are used to determine it is 1,2, or 3D
startPoint=[0.01880658333, -0.01164166666582, -5E-3];
endPoint=[0.1101, 0.0762, 0.0106];

%%%%%%%%% Nothing below here needs to be changed
matLabFiles = dir('*.mat');
for i=1:length(matLabFiles)
    % Get the file information
    matLabFile = matLabFiles(i).name;
    rawFileName= matLabFile(1:end-4);
    chrestFile = append(rawFileName, '.h5');
    newFile = ~ isfile(chrestFile);

    % load the matlab data 
    matStruct=load(matLabFile);

    % march over each field in the matStruct
    fields = fieldnames(matStruct);

    % for each field
    for k=1:numel(fields)
        % make sure the field is data/number type
        if( isnumeric(matStruct.(fields{k})) )
            % Get the specific matlab mat
            mat = vertcat(matStruct.(fields{k}));

            % Compute delta x y z based upon the field data
            delta = minus(endPoint, startPoint);
            matSize=zeros(1,numel(delta));
            for d=1:numel(delta)
               delta(d) = delta(d)/size(mat,d);
               matSize(d) = size(mat,d);
            end

            % Flip x to point in the same direction as numerical results
            mat = flip(mat, 1);

            % compute the field name
            hdf5FieldName = ['/data/fields/' fields{k}];
            
            % Write the data
            if newFile
                h5create(chrestFile,hdf5FieldName,matSize);
            end
            h5write(chrestFile,hdf5FieldName,mat);
        end
    end

    % create the grid information, it is assumed to be the same for all
    % fields
    if newFile
      h5create(chrestFile,'/data/grid/start',[1 numel(delta)]);
      h5create(chrestFile,'/data/grid/end',[1 numel(delta)]);
      h5create(chrestFile,'/data/grid/discretization',[1 numel(delta)]);
    end

    % write the grid data, this may override what we have, but it
    % should be the same
    h5write(chrestFile,'/data/grid/start',startPoint);
    h5write(chrestFile,'/data/grid/end',endPoint);
    h5write(chrestFile,'/data/grid/discretization',delta);
    % Write the metadata
    metadataKeys=keys(metadata);
    for m = 1:length(metadataKeys)
        key = metadataKeys{m};
        value = metadata(key);
        h5writeatt(chrestFile, '/data', key, value)
    end

    % Compute and save the current time
    indexString  = regexp(rawFileName, '(\d+)(?!.*\d)', 'match');
    index=str2double(indexString);
    time = timeStart+index*snapShotDt;
    
    % write the time into the data
    h5writeatt(chrestFile, '/data', 'time', time)

    
end