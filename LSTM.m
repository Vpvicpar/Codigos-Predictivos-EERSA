%% 1. Cargar y Preprocesar Datos
file_path = 'E:/EERSA/datoscsv.xlsx';
sheet_name = 'datoscsv';
data = readtable(file_path, 'Sheet', sheet_name);

% Convertir Fecha y Hora a formato datetime
Fechas = datetime(data.Fecha, 'InputFormat', 'yyyy-MM-dd');
Horas  = datetime(data.Hora,  'InputFormat', 'HH:mm:ss');
TimeTotal = Fechas + timeofday(Horas);  % Línea de tiempo completa

% Variables de interés
Consumo     = data.Consumo;
Temperatura = data.Temperatura;
DiaNum      = data.DiaNum;
Condicion   = data.Condicion;

% Características derivadas de la fecha y hora
HoraDelDia   = hour(Horas);
MinutoDelDia = hour(Horas)*60 + minute(Horas);
Mes          = month(Fechas);
SemanaDelAno = week(Fechas);
FinDeSemana  = ismember(weekday(Fechas), [1, 7]);

% Matriz de variables exógenas
X = [Temperatura, DiaNum, Condicion, HoraDelDia, MinutoDelDia, Mes, SemanaDelAno, FinDeSemana];

%% 2. Preparar Secuencias para el Entrenamiento del LSTM
lag = 4;  % Número de pasos de tiempo previos a considerar
numObservaciones = length(Consumo);
numExo = size(X, 2);
numFeatures = 1 + numExo;  % 1 para Consumo + n para las variables exógenas

inputs = {};
targets = {};
timeInputs = {};

for i = lag:numObservaciones-1
    % Cada secuencia tendrá tamaño: (numFeatures x lag)
    seqInput = [Consumo(i-lag+1:i)';      % Consumo: (1 x lag)
                X(i-lag+1:i, :)'];         % Exógenas: (numExo x lag)
    % Objetivo: consumo en el siguiente paso
    seqTarget = Consumo(i+1);
    
    inputs{end+1} = seqInput;
    targets{end+1} = seqTarget;
    timeInputs{end+1} = TimeTotal(i+1);
end

XSequences = inputs;    % Secuencias de entrada
YTargets   = targets;   % Valores objetivo
TimeTargets = timeInputs;

%% 3. Dividir en Entrenamiento, Validación y Prueba (60/20/20)
numSeq = numel(XSequences);
idxTrain = 1:floor(0.6*numSeq);
idxVal   = floor(0.6*numSeq)+1 : floor(0.8*numSeq);
idxTest  = floor(0.8*numSeq)+1 : numSeq;

% Conjunto de entrenamiento
XTrain = XSequences(idxTrain);
YTrain = YTargets(idxTrain);
TimeTrain = TimeTargets(idxTrain);

% Conjunto de validación
XVal = XSequences(idxVal);
YVal = YTargets(idxVal);
TimeVal = TimeTargets(idxVal);

% Conjunto de prueba
XTest = XSequences(idxTest);
YTest = YTargets(idxTest);
TimeTest = TimeTargets(idxTest);

%% 4. Normalización de Datos
% Calcular media y desviación utilizando únicamente el conjunto de entrenamiento
allTrainData = cell2mat(XTrain);
mu = mean(allTrainData, 2);
sigma = std(allTrainData, 0, 2);

normFunc = @(x) (x - mu)./sigma;
denormFunc = @(x) x.*sigma + mu;

% Normalizar el conjunto de entrenamiento
for i = 1:numel(XTrain)
    XTrain{i} = normFunc(XTrain{i});
end
% Normalizar el conjunto de validación
for i = 1:numel(XVal)
    XVal{i} = normFunc(XVal{i});
end
% Normalizar el conjunto de prueba
for i = 1:numel(XTest)
    XTest{i} = normFunc(XTest{i});
end

% Normalización de los objetivos (Consumo) utilizando la media y desviación global del consumo
muConsumo = mean(Consumo);
sigmaConsumo = std(Consumo);
YTrainNorm = cellfun(@(y) (y - muConsumo)/sigmaConsumo, YTrain);
YValNorm   = cellfun(@(y) (y - muConsumo)/sigmaConsumo, YVal);

%% 5. Definir y Entrenar la Red LSTM
numHiddenUnits = 50;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer];

% Convertir las respuestas normalizadas a un vector columna
YTrainNormMat = YTrainNorm(:);
YValNormMat   = YValNorm(:);

% Configurar las opciones de entrenamiento incluyendo datos de validación
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress', ...
    'ValidationData', {XVal, YValNormMat}, ...
    'ValidationFrequency', 10);

% Entrenar la red LSTM
netLSTM = trainNetwork(XTrain, YTrainNormMat, layers, options);

%% 6. Predicción y Evaluación en el Conjunto de Prueba
YPredNorm = predict(netLSTM, XTest);
YPred = YPredNorm * sigmaConsumo + muConsumo;

YTestMat = cell2mat(YTest);
MAPE_test = mean(abs((YTestMat - YPred) ./ YTestMat)) * 100;
RMSE_test = sqrt(mean((YTestMat - YPred).^2));
MAE_test  = mean(abs(YTestMat - YPred));

fprintf('Conjunto de Prueba - Error MAPE: %.2f%%\n', MAPE_test);
fprintf('Conjunto de Prueba - Error RMSE: %.2f\n', RMSE_test);
fprintf('Conjunto de Prueba - Error MAE: %.2f\n', MAE_test);

%% 7. Predicción y Evaluación en el Conjunto de Validación
YPredValNorm = predict(netLSTM, XVal);
YPredVal = YPredValNorm * sigmaConsumo + muConsumo;

YValMat = cell2mat(YVal);
MAPE_val = mean(abs((YValMat - YPredVal) ./ YValMat)) * 100;
RMSE_val = sqrt(mean((YValMat - YPredVal).^2));
MAE_val  = mean(abs(YValMat - YPredVal));

fprintf('Conjunto de Validación - Error MAPE: %.2f%%\n', MAPE_val);
fprintf('Conjunto de Validación - Error RMSE: %.2f\n', RMSE_val);
fprintf('Conjunto de Validación - Error MAE: %.2f\n', MAE_val);

%% 8. Gráfica Global: Predicción vs Dato Real (Conjunto de Prueba)
TimeTestArray = [TimeTest{:}];
n_test = min([length(TimeTestArray), length(YTestMat), length(YPred)]);
TimeTestArray = TimeTestArray(1:n_test);
YTestMat = YTestMat(1:n_test);
YPred = YPred(1:n_test);

figure;
plot(TimeTestArray, YTestMat, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Dato Real');
hold on;
plot(TimeTestArray, YPred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicción');
xlabel('Tiempo (HH:mm)');
ylabel('Consumo');
title('Comparación Global: Dato Real vs Predicción (LSTM) - Conjunto de Prueba');
grid on;
legend('Location','best');
xtickformat('HH:mm');
saveas(gcf, 'comparacion_global_LSTM.png');

%% 9. Creación de la Tabla de Resultados para el Conjunto de Prueba
n_test = min([length(TimeTestArray), length(YTestMat), length(YPred)]);
TimeTestArray = TimeTestArray(1:n_test);
YTestMat = YTestMat(1:n_test);
YPred = YPred(1:n_test);

resultTable = table(TimeTestArray, YTestMat, YPred, ...
    'VariableNames', {'FechaHora', 'DatoReal', 'Prediccion'});
disp(resultTable);
writetable(resultTable, 'resultados_prueba_LSTM.xlsx');
% Guardar variables necesarias
save('resultados_LSTM.mat', 'TimeTestArray', 'YTestMat', 'YPred');

%% 10. (Opcional) Visualización de un Día Aleatorio del Conjunto de Prueba
% Asegurar que las variables necesarias están en el Workspace:
% TimeTestArray: Fechas y horas de los datos de prueba
% YTestMat: Consumo real correspondiente
% YPred: Consumo predicho correspondiente

% Obtener todos los días únicos en el conjunto de prueba
diasUnicos = unique(dateshift(TimeTestArray, 'start', 'day'));

% Seleccionar un día aleatorio
rng('shuffle'); % Para asegurar aleatoriedad en cada ejecución
diaAleatorio = datasample(diasUnicos, 1);

% Filtrar los datos correspondientes al día seleccionado
indicesDia = (dateshift(TimeTestArray, 'start', 'day') == diaAleatorio);
horasDia = timeofday(TimeTestArray(indicesDia)); % Extraer solo la hora del día
consumoReal = YTestMat(indicesDia); % Obtener el consumo real para ese día
consumoPredicho = YPred(indicesDia); % Obtener el consumo predicho para ese día

% Convertir horasDia a datetime con una fecha ficticia (ej. '01-Jan-2000')
horasDiaDatetime = datetime(2000,1,1) + horasDia;

% Graficar el consumo real y predicho
figure;
plot(horasDiaDatetime, consumoReal, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Consumo Real');
hold on;
plot(horasDiaDatetime, consumoPredicho, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Consumo Predicho');
hold off;

xlabel('Hora del día');
ylabel('Consumo');
title(['Consumo en intervalos de 15 minutos - Día: ', datestr(diaAleatorio, 'yyyy-mm-dd')]);
grid on;
legend('Location', 'best'); % Agregar leyenda
datetick('x', 'HH:MM', 'keeplimits'); % Formatear el eje X

% Mostrar el día seleccionado en la consola
fprintf('Día seleccionado aleatoriamente: %s\n', datestr(diaAleatorio, 'yyyy-mm-dd'));

% (Opcional) Guardar la figura
% saveas(gcf, 'grafica_dia_aleatorio.png');
