% Purpose : Split data into training and testing using festival train test method

wavpath = '../../../feats/lmag/';
files = dir(strcat(wavpath,'*.lmag'));

text_fulldir = '../../../feats/lmag/';
audio_fulldir = '../../../feats/irm/';

text_traindir = '../../../data/train/lmag/';
audio_traindir = '../../../data/train/irm/';

text_valdir = '../../../data/val/lmag/';
audio_valdir = '../../../data/val/irm/';

text_testdir = '../../../data/test/lmag/';
audio_testdir = '../../../data/test/irm/';

mkdir(text_traindir);
mkdir(text_valdir);
mkdir(text_testdir);

mkdir(audio_traindir);
mkdir(audio_valdir);
mkdir(audio_testdir);

aext = '.irm';
text = '.lmag';

ivecv = [1:9:length(files)];
ivect = [1:10:length(files)];

for i = 1:length(files)
    
    [fname,tok] = strtok(files(i).name,'.');
    fprintf('processing utt %s for splitting data into train/test/val \n',fname);
    
    if isempty(find(ivecv == i, 1))
        if ( exist(strcat(audio_fulldir,fname,aext), 'file') && exist(strcat(text_fulldir,fname,text), 'file') )
            command = ['cp', ' ',strcat(audio_fulldir,fname,aext), ' ', strcat(audio_traindir,fname,aext)];
            system(command);
            command = ['cp', ' ',strcat(text_fulldir,fname,text), ' ', strcat(text_traindir,fname,text)];
            system(command);
        end
    else if isempty(find(ivect == i, 1))
            if ( exist(strcat(audio_fulldir,fname,aext), 'file') && exist(strcat(text_fulldir,fname,text), 'file') )
                command = ['cp', ' ',strcat(audio_fulldir,fname,aext), ' ', strcat(audio_valdir,fname,aext)];
                system(command);
                command = ['cp', ' ',strcat(text_fulldir,fname,text), ' ', strcat(text_valdir,fname,text)];
                system(command);
            end
        else
            if ( exist(strcat(audio_fulldir,fname,aext), 'file') && exist(strcat(text_fulldir,fname,text), 'file') )
                command = ['cp', ' ',strcat(audio_fulldir,fname,aext), ' ', strcat(audio_testdir,fname,aext)];
                system(command);
                command = ['cp', ' ',strcat(text_fulldir,fname,text), ' ', strcat(text_testdir,fname,text)];
                system(command);
            end
        end
    end
    
end
