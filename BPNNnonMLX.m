tic;
% Membaca dataset
data = readtable('iris.data.csv');
input = table2array(data(:, 1:end-1));
labels = data{:, end};

% Konversi label teks menjadi angka
uniqueLabels = unique(labels);
target = zeros(height(data), 1);
for i = 1:length(uniqueLabels)
    target(strcmp(labels, uniqueLabels{i})) = i;
end

% Normalisasi data
input = (input - min(input)) ./ (max(input) - min(input));

% Inisialisasi
alphas = 0.1:0.2:0.7;
mius = 0.6:0.1:0.8;
ratios =  0.1:0.2:0.5;
iterations = 10;

% Inisialisasi tabel untuk menyimpan hasil
results = table();
totalTraining = length(alphas) * length(mius) * length(ratios) * iterations;
currentTraining = 0;


for alpha = alphas
    for miu = mius
        for ratio = ratios
            epochList = zeros(1, iterations);
            accuracyList = zeros(1, iterations);
            
            for iter = 1:iterations
                % Membagi dataset
                % Membagi dataset menjadi training dan testing dengan Stratified Sampling
                uniqueTargets = unique(target);
                trainIdx = [];
                testIdx = [];
                
                for i = 1:length(uniqueTargets)
                    classIdx = find(target == uniqueTargets(i)); % Indeks dari kelas tertentu
                    numTrainClass = round(ratio * length(classIdx));
                    
                    randClassIdx = randperm(length(classIdx));
                    
                    trainClassIdx = classIdx(randClassIdx(1:numTrainClass));
                    testClassIdx = classIdx(randClassIdx(numTrainClass+1:end));
                    
                    trainIdx = [trainIdx; trainClassIdx];
                    testIdx = [testIdx; testClassIdx];
                end
                
                inputTrain = input(trainIdx, :);
                targetTrain = target(trainIdx);
                inputTest = input(testIdx, :);
                targetTest = target(testIdx);

                % Inisialisasi
                n = size(inputTrain, 2);
                m = 10;
                l = length(unique(targetTrain));
                max_epoch = 10000;
                EP = inf;

                % Inisialisasi bobot dan bias dengan Nguyen-Widrow
                beta = 0.7 * m^(1/n);
                v = rand(n, m) * 2 - 1;
                normV = sqrt(sum(v.^2));
                v = beta * v ./ normV;
                v0 = (rand(1, m) * 2 - 1) * beta;
                w = rand(m, l) * 2 - 1;
                w0 = rand(1, l) * 2 - 1;

                deltaWPrev = zeros(size(w));
                deltaVPrev = zeros(size(v));
                deltaW0Prev = zeros(size(w0));
                deltaV0Prev = zeros(size(v0));

                epoch = 0;

                while EP > 0.01 && epoch < max_epoch
                    EP = 0;
                    for p = 1:size(inputTrain, 1)
                        % (kode feedforward dan backpropagation)
                        for p = 1:size(inputTrain, 1)
                            xi = inputTrain(p, :);
                            tk = zeros(1, l);
                            tk(targetTrain(p)) = 1;
                        
                            % Feedforward
                            z_in = zeros(1, m);
                            z = zeros(1, m);
                            for j = 1:m
                                z_in(j) = xi * v(:, j) + v0(j);
                                z(j) = 1 / (1 + exp(-z_in(j)));
                            end
                            
                            y_in = zeros(1, l);
                            y = zeros(1, l);
                            for k = 1:l
                                y_in(k) = z * w(:, k) + w0(k);
                                y(k) = 1 / (1 + exp(-y_in(k)));
                            end
                            
                            % Backpropagation
                            delta_k = (tk - y) .* y .* (1 - y);
                            delta_j = (delta_k * w') .* z .* (1 - z);
                        
                            % Update bobot dan bias
                            for j = 1:m
                                for i = 1:n
                                    v(i, j) = v(i, j) + alpha * xi(i) * delta_j(j);
                                end
                                v0(j) = v0(j) + alpha * delta_j(j);
                            end
                        
                            for k = 1:l
                                for j = 1:m
                                    w(j, k) = w(j, k) + alpha * z(j) * delta_k(k);
                                end
                                w0(k) = w0(k) + alpha * delta_k(k);
                            end
                        
                            EP = EP + sum((tk - y).^2)/2;
                        end

                        EP = EP + sum((tk - y).^2)/2;
                    end
                    EP = EP / size(inputTrain, 1);
                    epoch = epoch + 1;
                end

                % Evaluasi pada dataset testing
                   correct = 0;
                for p = 1:size(inputTest, 1)
                    % (kode feedforward untuk evaluasi)
                    xi = inputTest(p, :);
                    tk = zeros(1, l);
                    tk(targetTest(p)) = 1;
                
                    % Feedforward
                    z_in = zeros(1, m);
                    z = zeros(1, m);
                    for j = 1:m
                        z_in(j) = xi * v(:, j) + v0(j);
                        z(j) = 1 / (1 + exp(-z_in(j)));
                    end
                
                    y_in = zeros(1, l);
                    y = zeros(1, l);
                    for k = 1:l
                        y_in(k) = z * w(:, k) + w0(k);
                        y(k) = 1 / (1 + exp(-y_in(k)));
                    end
                
                    [~, predicted] = max(y);
                    if predicted == targetTest(p)
                        correct = correct + 1;
                    end
                end
                
                accuracy = correct / size(inputTest, 1) * 100;
                
                epochList(iter) = epoch;
                accuracyList(iter) = accuracy;
                currentTraining = currentTraining + 1;
                elapsedTime = toc; % Dapatkan waktu yang telah berlalu
                fprintf('Alpha: %f, Miu: %f, Ratio: %f, Iteration: %d, Training Completed: %d/%d, Elapsed Time: %.2f seconds\n', alpha, miu, ratio, iter, currentTraining, totalTraining, elapsedTime);
            end
            
           % Menghitung statistik deskriptif
            meanEpoch = mean(epochList);
            stdEpoch = std(epochList);
            meanAccuracy = mean(accuracyList);
            stdAccuracy = std(accuracyList);
            
            % Menambahkan hasil ke tabel
            newRow = {alpha, miu, ratio, meanEpoch, stdEpoch, meanAccuracy, stdAccuracy};
            results = [results; newRow];
        end
    end
end


% Menamai kolom tabel
results.Properties.VariableNames = {'Alpha', 'Miu', 'Ratio', 'MeanEpoch', 'StdEpoch', 'MeanAccuracy', 'StdAccuracy'};

% Menampilkan tabel
disp(results);

elapsed_time = toc;  % Menghentikan penghitungan waktu dan menyimpannya ke variabel elapsed_time

fprintf('Waktu eksekusi: %f detik\n', elapsed_time);
