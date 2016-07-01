% Sanjit Singh Batra: MCMR v2.1: 29-01-2015

%function []= MCMR_kernel(xTrain, yTrain, xTest, yTest)


%     xTrain = csvread('xTrain.txt');
%     yTrain = csvread('yTrain.txt');
%     xTest = csvread('xTest.txt');
%     yTest = csvread('yTest.txt');
    
    
    %clear;
    clc;
    
    tic
    
    C = 1e0;
    beta = 0.1;
    epsilon = 0.001;

    %RBF 
    Kernel = @(x,y,beta) exp(-beta * norm(x-y)^2);
    
    %Linear
%     Kernel = @(x,y,beta) ( (x*y'));

    
    N=size(xTrain,1);
    D=size(xTrain,2);
    
    
    %%Normalizing values 
    
    % Normalize x_values
    m = mean(xTrain);
    stdev = std(xTrain);
    for d=1:D
        if(stdev(d)~=0)
        xTrain(:,d) = (xTrain(:,d) - m(d))/stdev(d);
        xTest(:,d) = (xTest(:, d) - m(d))/stdev(d);
        else
            xTrain(:,d) = (xTrain(:,d) - m(d));
            xTest(:,d) = (xTest(:, d) - m(d));
        end          
    end
    
    
        %[lambda   ;b         ; Q+       ; Q-       ;H         ;P  ]
    X = [randn(N,1);randn(1,1);randn(N,1);randn(N,1);randn(1,1);randn(1,1)];       %[lambda,b,q+,q-,H,P]
    f = [zeros(N,1);zeros(1,1);C*ones(N,1);C*ones(N,1);ones(1,1);zeros(1,1)];
    
    eps_1 = yTrain + epsilon*(ones(N,1));%(Yi + epsilon)
    eps_2 = yTrain - epsilon*(ones(N,1)) ;% ( Yi - epsilon)


    A = zeros(4*N, 3*N + 3);
    b = zeros(4*N, 1);


    %  implementing (lambda)*K(xi,xj) +b +p(Yi + epsilon) - H <= 0 
    for i = 1:N
    for j = 1:N
        atemp = Kernel(xTrain(i, :), xTrain(j, :), beta);
        A(i, j) = atemp;% lambda
    end
    A(i,  N+1) = 1; %b
    A(i, N+1 + i) = 0;%q+ 
    A(i,2*N+1 + i) = 0;%q-
    A(i,3*N + 2) = -1;%H
    A(i,3*N + 3) = eps_1(i);%P
    b(i,1) = 0;
    end

      %  implementing -(lambda)*K(xi,xj) -b -p(Yi - epsilon) -H <= 0 

    offset = N;
    for i = 1:N
    for j = 1:N
        atemp = Kernel(xTrain(i, :), xTrain(j, :), beta);
        A( offset + i, j) = -atemp;%lambda
    end
    A(offset + i,N+1) = -1; %b
    A(offset + i,N+1 + i) = 0;%q+
    A(offset + i,2*N+1 + i) = 0;%q- 
    A(offset + i,3*N + 2) = -1;%H
    A(offset + i,3*N + 3) = -1*eps_2(i);%P
    b(offset + i,1) = 0;
    end

      %  implementing -(lambda)*K(xi,xj) - b - p(y + epsilon) - q+  <= -1 


    offset = 2*N;
    for i = 1:N
    for j = 1:N
        atemp = Kernel(xTrain(i, :), xTrain(j, :), beta);
        A( offset + i, j) = -atemp;%lambda
    end
    A(offset + i,N+1) = -1; %b
    A(offset + i,N+1 + i) = -1;%q+
    A(offset + i,2*N+1 + i) = 0;%q-
    A(offset + i,3*N + 2) = 0;%H
    A(offset + i,3*N + 3) = -1*eps_1(i);%P
    b(offset + i,1) = -1;
    end

      %  implementing (lambda)*K(xi,xj) + b + p(y - epsilon) - q-  <= -1 

       offset = 3*N;
    for i = 1:N
    for j = 1:N
        atemp = Kernel(xTrain(i, :), xTrain(j, :), beta);
        A( offset + i, j) = atemp;%lambda
    end
    A(offset + i,N+1) =  1; %b
    A(offset + i,N+1 + i) = 0;%q+
    A(offset + i,2*N+1 + i) = -1;%q-
    A(offset + i,3*N + 2) = 0;%H
    A(offset + i,3*N + 3) = eps_2(i);%P
    b(offset + i,1) = -1;
    end 





    Aeq = [];
    beq = [];

    lb = [-inf*ones(N,1);-inf*ones(1,1);0*ones(N,1);0*ones(N,1);0;-inf];
    ub = [ inf*ones(N,1); inf*ones(1,1);inf*ones(N,1);inf*ones(N,1);inf;inf];
    options=optimset('display','none', 'Largescale', 'off', 'Simplex', 'on');

    [X,fval,exitflag] = linprog(f,A,b,Aeq,beq,lb,ub,X,options);
%     fprintf(2,'H is %f, fval is %f\n',full(X(3*N+2)),full(fval));
    
    if(exitflag ~= 1)
        fprintf('Please choose suitable values of C and beta\n');
        return;
    end
    
    lambda = X(1:N,:);

    H = full(X(3*N+2,:));
    P = full(X(3*N+3,:));
    lambda = lambda;
    bias = X(N+1,:);
    ntest = size(xTest,1);
    yPred = zeros( size(xTest,1),1);

    for i = 1:ntest
    sum = bias;
    for j = 1:N
        sum = sum + (lambda(j)) * Kernel(xTrain(j, :), xTest(i, :), beta);
    end
    yPred(i) = (-1/P)*(sum);
    end
    testlse = norm(yPred-yTest)^2/length(yTest);


    nsv = 0;
    for i = 1: N

        if((lambda(i))~=0)
            nsv = nsv + 1;
        end 
    end
    
    H
    lambda
    bias
    [yPred, yTest]
    
    fprintf(2, 'P is %f\nMSE = %f\nnsv = %d\n\n',P,testlse,nsv);
    
    toc
    
   %LibSVM
    model = svmtrain2(yTrain,xTrain,'-s 3 -t 0 -q');
    [pred,LibSVMacc,prob] = svmpredict(yTest,xTest,model,'-q');
    testlse = norm(pred-yTest)^2/length(yTest);

    [pred, yTest]
    
    fprintf(2, '\nLibSVM MSE = %f \nnsv = %d \n\n', testlse,model.totalSV);

%end



