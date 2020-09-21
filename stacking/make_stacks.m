%%% Event stacking for the code of our CVPR paper: E2SRI "Learning to
%%% Super-Resolve Intensity Images from Events"
%%% S. Mohammad Mostafavi I., Yeongwoo Nam, Jonghyun Choi and Kukjin Yoon.
%%% https://github.com/gistvision/e2sri
%%%
%%% Written by S. Mohammad Mostafavi I.

clear;% clc;
format longG

file_name='slider_depth';
load_folder = '/home/mohammad/e2sri/'; % change to user name
save_folder = '/home/mohammad/e2sri/'; % change to user name
addpath(genpath('matlab_rosbag-0.5.0-linux64')); % https://github.com/bcharrow/matlab_rosbag/releases
loc=strcat(save_folder,file_name,'/');
mkdir(loc)
tic
fprintf('Reading from ROS bag named: %s\n', file_name);
Bag = ros.Bag.load(strcat(load_folder, file_name, '.bag'));
[a, b] = Bag.readAll('/dvs/image_raw');
[c, ~] = Bag.readAll('/dvs/events');
fprintf('Done reading! >>> ')
toc

%%%%% Start memory managment of large file
all_events=[]; event_split=[];
split=1:50:size(c,2);
split(size(split,2)+1)=size(c,2);
fprintf('Reading %d Events from ROS as %d packages. >>> ', size(c,2),(size(split,2)-1))
for j=1:(size(split,2)-1)
    for i=split(j):split(j+1)
        event_split=cat(2,event_split,c{1, i}.events);
        %i
    end
    all_events=cat(2,all_events,event_split);
    event_split=[];
end
toc
%%%%% End memory managment

% Uncomment this line (and L62,L63) if you also need the time-stamps
%fid =fopen(strcat(loc,'image_timestamps.txt'), 'a' );

frame_number=1;

% These two numbers change the output quality heavily
% stack_shift NOT used these numbers when synchromizimg with the APS frames
% The location of APS frame is used and 4 frames before that plus 3 frames 
% after will make the correct sequence for the 7S example (will release)
events_per_stack=5000; % Number of events per frame
stack_shift=14000;     % Number of events between two consecutive stacks

while size(all_events,2)>=(3*events_per_stack)

        event_frame=uint8(ones(240,180).*128); % empty frame
        event_stack=cat(3,event_frame,event_frame,event_frame); % empty stack
        
        event_stream_stack=all_events(:, (1:3*events_per_stack)); % chunk of the stream to stack
        
        % fill events inside each frame
        for k=1:3
                    
            for j=((k-1)*events_per_stack+1):(k*events_per_stack)
                event_stack (event_stream_stack(1,j)+1,event_stream_stack(2,j)+1,k) = event_stream_stack(4,j).*256;
            end
            
        end
        
        event_stack=fliplr(rot90(event_stack,3)); % correct the orientation
        imwrite(event_stack,strcat(loc,sprintf('%04d',(frame_number)),'.png')) % write to file
       
    %TS(number)=EE(3,numE+(numE/2));
    %fprintf(fid, '%f\n', TS(number));
   
    frame_number=frame_number+1
    
    all_events=all_events(:,stack_shift:end);
    
end

%fclose(fid);
