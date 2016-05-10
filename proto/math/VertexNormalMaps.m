%This implementation does not have bilateral filtering so a noisy
%reconstruction is to be expected. 

% read a depth image
d = imread('frame-000000.depth.png');

% Kinect V1 camera matrix
K = [ 
    585      0   320;
      0   -585   240;
      0      0     1;
    ];

% allocate Vertex Map and calc
V = zeros(480,640,3);
for i = 1:480
    for j = 1:640
        V(i,j,:) = double(d(i,j))*inv(K)*[i j 1]';
    end
end

% allocate Normal Map and calc
N = zeros(480, 640, 3);
for i = 1:479
    for j = 1:639
        n = cross(V(i+1,j,:) - V(i,j,:), V(i,j+1,:) - V(i,j,:));
        n = n(:,:);
        N(i,j,:) = n/norm(n);
    end
end

% matlab makes point clouds easy! Map point color to normal
ptCloud = pointCloud(V, 'Color', uint8(N*255));
pcshow(ptCloud);