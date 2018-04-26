function  mean_error = motionGenerationError( dirname,iter_use )

clrs='rgbkmc';

if nargin < 2
    iter_use = 4000;
end;

legend_to_add = {};
fnum = 1;
figure;
toplot = false;
mean_error = [];
error_5 = [];
error_5_val = 0;
iter = 1;
it = [];
while(1)
    % for iteration =   iter_use
    R0 = eye(3);
    T0 = [0 0 0];
    errors = [];
    for N = 0:7
        if exist([dirname,'/test_ground_truth_unnorm_N_', num2str(N) ],'file') ~= 2
            disp([dirname,'/test_ground_truth_unnorm_N_', num2str(N) ])
            disp('BOOOOOOOM');
            continue
        end;
        try 
            f=csvread([dirname,'/test_ground_truth_unnorm_N_', num2str(N)]);
            fstd = std(f,1);
            idx_to_use = find(fstd>1e-4);

            expchannels = revertCoordinateSpace(f,R0,T0);
            eulerchannels = expchannels;
            for i = 1:size(expchannels,1)
                for j = 4:3:97
                   eulerchannels(i,j:j+2) =  RotMat2Euler(expmap2rotmat(expchannels(i,j:j+2)));
                end;
            end;
            eulerchannels(:,1:6) = 0;
            fstd = std(eulerchannels,1);
            idx_to_use = find(fstd>1e-4);

            
            if exist([dirname,'/forecast_iteration_unnorm_N_',num2str(N),],'file') ~= 2
                disp('kaBOOOOOOOM');
            continue
            end;  
            try      
                f=csvread([dirname,'/forecast_iteration_unnorm_N_',num2str(N)]);
                expchannels = revertCoordinateSpace(f,R0,T0);
                eulerchannels_forecast = expchannels;
                for i = 1:size(expchannels,1)
                    for j = 4:3:97
                       eulerchannels_forecast(i,j:j+2) =  RotMat2Euler(expmap2rotmat(expchannels(i,j:j+2)));
                    end;
                end;

                err = (eulerchannels(:,idx_to_use) - eulerchannels_forecast(:,idx_to_use)).^2;
                v=sum(err,2);
                errors(:,N+1) = sqrt(v);
            catch er
                continue
            end
        catch er
            continue
        end
    end;
    % if size(errors,1) > 0
    %toplot = true;
    error_5_val_old = error_5_val;
    mean_error = mean(errors,2);
    error_5_val = mean_error(5);
    % legend_to_add{fnum} = ['iteration = ',num2str(iteration)];
    if fnum <= size(clrs,2)
        clr = clrs(fnum);
    else
        clr = rand(1,3);
    end;
    if(error_5_val~=error_5_val_old)
        txt = sprintf('Errors := %.2f, %.2f, %.2f, %.2f, %.2f', mean_error(8), mean_error(16), mean_error(32), mean_error(56), mean_error(100));
	disp(txt);
	%error_5(iter) = error_5_val
        %it(iter) = iter;
        %scatter(it,error_5,'filled')%,'color',clr,'linewidth',3);
        iter = iter + 1;
    end;
    % hold on;
    pause(10)
    % fnum = fnum + 1;
    % end;
    
    % if toplot
    %     l=legend(p,legend_to_add);
    %     set(l,'FontSize',20)
    % end
    end

end
