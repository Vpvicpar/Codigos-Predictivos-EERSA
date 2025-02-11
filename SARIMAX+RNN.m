%% Cargar datos desde Excel
file_path = 'E:/EERSA/datoscsv.xlsx';
sheet_name = 'datoscsv';
data = readtable(file_path, 'Sheet', sheet_name);

%% Convertir Fecha y Hora a formato datetime y extraer variables
Fechas = datetime(data.Fecha, 'InputFormat', 'yyyy-MM-dd'); % Ajustar formato si es necesario
Horas  = datetime(data.Hora,  'InputFormat', 'HH:mm:ss');    % Ajustar formato si es necesario

% Para la línea de tiempo completa, combinamos la fecha con la hora.
TimeTotal = Fechas + timeofday(Horas);

% Extraer características de fecha y hora
HoraDelDia   = hour(Horas);                     % 0 a 23
MinutoDelDia = hour(Horas)*60 + minute(Horas);    % 0 a 1439
Mes          = month(Fechas);                   % 1 a 12
SemanaDelAno = week(Fechas);                     % 1 a 52
FinDeSemana  = ismember(weekday(Fechas), [1, 7]); % 1 si sábado o domingo, 0 si no

% Extraer variables de interés
Consumo     = data.Consumo;
Temperatura = data.Temperatura;
DiaNum      = data.DiaNum;
Condicion   = data.Condicion;

% Matriz de variables exógenas
X = [Temperatura, DiaNum, Condicion, HoraDelDia, MinutoDelDia, Mes, SemanaDelAno, FinDeSemana];

%% División de datos (80% entrenamiento, 20% prueba)
n = length(Consumo);
train_ratio = 0.8;
n_train = floor(n * train_ratio);
n_test  = n - n_train;

Y_train = Consumo(1:n_train);
Y_test  = Consumo(n_train+1:end);
X_train = X(1:n_train, :);
X_test  = X(n_train+1:end, :);
Time_train = TimeTotal(1:n_train);
Time_test  = TimeTotal(n_train+1:end);

%% MODELO SARIMAX
% Se utiliza un modelo ARIMA sin términos autoregresivos y con un término de media móvil
Mdl = arima('ARLags', [], 'D', 0, 'MALags', 1);
Mdl = estimate(Mdl, Y_train, 'X', X_train);

% Predicción para el conjunto de prueba
Y_pred_sarima = forecast(Mdl, n_test, 'X0', X_train, 'XF', X_test);

% Calcular los residuos en el conjunto de entrenamiento (diferencia entre valores reales y ajustados)
residuals = Y_train - infer(Mdl, Y_train, 'X', X_train);

%% NORMALIZACIÓN PARA LA RED NEURONAL
% Se normalizan las variables exógenas (entradas) y los residuos (salida de la red)
[Xn, Xs] = mapminmax(X_train');        % Normaliza entradas
[Yn, Ys] = mapminmax(residuals');        % Normaliza residuos

%% RED NEURONAL PARA PREDECIR LOS RESIDUOS
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize, 'trainlm'); % Red MLP con 10 neuronas ocultas
net.trainParam.epochs = 100;
net.trainParam.min_grad = 1e-6;
net = train(net, Xn, Yn);

% Predicción de los residuos para el conjunto de prueba
Xn_test = mapminmax('apply', X_test', Xs);
residuals_pred_norm = net(Xn_test);
residuals_pred = mapminmax('reverse', residuals_pred_norm, Ys);

%% PREDICCIÓN FINAL COMBINADA
% Se suman las predicciones del modelo SARIMAX con la predicción de residuos de la red neuronal
Y_pred_final = Y_pred_sarima + residuals_pred';
%% PREDICCIÓN FINAL COMBINADA
Y_pred_final = Y_pred_sarima + residuals_pred';

% Calcular y mostrar errores
MAE = mean(abs(Y_test - Y_pred_final));
disp(['Error Absoluto Medio (MAE): ', num2str(MAE)]);

RMSE = sqrt(mean((Y_test - Y_pred_final).^2));
disp(['Error Cuadrático Medio (RMSE): ', num2str(RMSE)]);

MAPE = mean(abs((Y_test - Y_pred_final) ./ Y_test)) * 100;
disp(['Porcentaje de Error Absoluto Medio (MAPE): ', num2str(MAPE), '%']);


%% (OPCIONAL) CORRECCIÓN DE SESGO
% Si observas un sesgo sistemático, calcula el sesgo y corrígelo.
bias = mean(residuals);  % Sesgo promedio en el conjunto de entrenamiento
% Descomenta la siguiente línea para aplicar la corrección de sesgo:
% Y_pred_final = Y_pred_final - bias;

%% GRÁFICA GLOBAL DE COMPARACIÓN (Predicho vs Real)
figure;
plot(Time_test, Y_test, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Real');
hold on;
plot(Time_test, Y_pred_final, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicción');
xlabel('Tiempo (HH:mm)');
ylabel('Consumo');
title('Comparación Global: Consumo Real vs Predicción');
grid on;
legend('Location', 'best');
% Formatear la línea de tiempo para mostrar horas y minutos
xtickformat('HH:mm');
saveas(gcf, 'comparacion_global.png');

%% VISUALIZACIÓN DE LOS RESIDUOS DEL MODELO (Conjunto de Prueba)
residuos_test = Y_test - Y_pred_final;
figure;
plot(Time_test, residuos_test, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Residuos');
xlabel('Tiempo (HH:mm)');
ylabel('Residuos');
title('Residuos del Modelo SARIMAX + Red Neuronal (Conjunto de Prueba)');
grid on;
legend('Location', 'best');
xtickformat('HH:mm');
saveas(gcf, 'residuos_global.png');

%% CREACIÓN DE LA TABLA DE RESULTADOS PARA EL CONJUNTO DE PRUEBA
% Se crea una tabla que incluye la fecha y hora, el dato real y la predicción para cada observación del conjunto de prueba.
resultTable = table(Time_test, Y_test, Y_pred_final, ...
    'VariableNames', {'FechaHora', 'DatoReal', 'Prediccion'});

% Mostrar la tabla en la consola
disp(resultTable);

% Guardar la tabla en un archivo Excel
writetable(resultTable, 'resultados_REDN.xlsx');




