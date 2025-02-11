% Cargar los datos desde un archivo Excel
datos_consumo_energia = readtable('E:\EERSA\datoscsv.xlsx');


% Supongamos que los datos están en una tabla con las columnas:
% 'Consumo', 'Temperatura', 'Dia', 'Condicion'

% Convertir variables categóricas de texto a variables dummy
%dummyDia = array2table(dummyvar(categorical(datos_consumo_energia.Dia)), ...
  %  'VariableNames', {'Dia_fs', 'Dia_s'}); % Fin de semana y entre semana
%dummyCondicion = array2table(dummyvar(categorical(datos_consumo_energia.Condicion)), ...
 %   'VariableNames', {'Condicion_no', 'Condicion_si'}); % No feriado y feriado

% Concatenar las variables dummy con la tabla original
datos_consumo_energia = [datos_consumo_energia, datos_consumo_energia.Dia, datos_consumo_energia.Condicion];

totalDatos = size(datos_consumo_energia, 1);

% Parámetros del modelo SARIMAX
p = 0;  % Número de lags autoregresivos (AR)
q = 0;  % Número de lags de media móvil (MA)
s = 0;  % Estacionalidad diaria (96 observaciones)
P = 0;  % Lags autoregresivos estacionales (SAR)
Q = 0;  % Lags de media móvil estacionales (SMA)
D = 1;  % Diferenciación estacional

% Número de predictores exógenos (en este caso, puedes ajustar según los datos que uses)
num_predictores_exogenos = 3;  % Ejemplo, supón que tienes 3 variables exógenas (como temperatura, día de la semana, feriado)

% Calcular el número mínimo de observaciones necesarias
observaciones_minimas = max([p, q, P, Q]) + s * D + num_predictores_exogenos;

% Mostrar el resultado
fprintf('El número mínimo de observaciones necesarias para este modelo es: %d\n', observaciones_minimas);
observaciones_minimas_vector=rand(observaciones_minimas, num_predictores_exogenos);


% Dividir los datos en conjuntos de entrenamiento, validación y prueba con más datos para entrenamiento
tamanoEntrenamiento = round(0.6 * totalDatos); % Aumentar el porcentaje de entrenamiento
tamanoValidacion = round(0.20 * totalDatos);
tamanoPrueba = totalDatos - tamanoEntrenamiento - tamanoValidacion;

datosEntrenamiento = datos_consumo_energia(1:tamanoEntrenamiento, :);
datosValidacion = datos_consumo_energia(tamanoEntrenamiento+1:tamanoEntrenamiento+tamanoValidacion, :);
datosPrueba = datos_consumo_energia(tamanoEntrenamiento+tamanoValidacion+1:end, :);

% Continuar con la definición de las variables de entrenamiento
yEntrenamiento = datosEntrenamiento.Consumo;
XEntrenamiento = [datosEntrenamiento.Temperatura, datosEntrenamiento.Dia, datosEntrenamiento.Condicion];


% Verificar las dimensiones de las variables
disp('Tamaño de yEntrenamiento:');
disp(size(yEntrenamiento));

disp('Tamaño de XEntrenamiento:');
disp(size(XEntrenamiento));

% Verificar que no haya valores NaN en los predictores
disp('Valores NaN en XEntrenamiento:');
disp(sum(isnan(XEntrenamiento)));

% Si hay valores NaN, eliminarlos
XEntrenamiento = rmmissing(XEntrenamiento);

% Verificar nuevamente las dimensiones
disp('Nuevo tamaño de XEntrenamiento después de eliminar NaN:');
disp(size(XEntrenamiento));
XEntrenamiento = table2array(datosEntrenamiento(:, {'Temperatura', 'Dia', 'Condicion'}));

% Definir y estimar el modelo ARIMA
mdl = arima('ARLags', [], 'D', 0, 'MALags', 1);
modelo = estimate(mdl, yEntrenamiento, 'X', XEntrenamiento);


% Validar el modelo usando el conjunto de validación
yValidacion = datosValidacion.Consumo;
XValidacion = [datosValidacion.Temperatura, datosValidacion.Dia, datosValidacion.Condicion];

% Pronosticar
[prediccionValidacion, ~] = forecast(modelo, length(yValidacion), 'Y0', yEntrenamiento, 'X0', XEntrenamiento, 'XF', XValidacion);

% Calcular el error de validación
errorValidacion = mean(abs(yValidacion - prediccionValidacion));

% Probar el modelo usando el conjunto de prueba
yPrueba = datosPrueba.Consumo;
XPrueba = [datosPrueba.Temperatura, datosPrueba.Dia, datosPrueba.Condicion];

% Pronosticar
[prediccionPrueba, ~] = forecast(modelo, length(yPrueba), 'Y0', yEntrenamiento, 'X0', XEntrenamiento, 'XF', XPrueba);

% Calcular el error de prueba
errorPrueba = mean(abs(yPrueba - prediccionPrueba));

% Mostrar los errores de validación y prueba
fprintf('Error de validación: %.2f\n', errorValidacion);
fprintf('Error de prueba: %.2f\n', errorPrueba);

% Gráfica de resultados
figure;
hold on;
plot(yPrueba, 'b', 'DisplayName', 'Consumo Real');
plot(prediccionPrueba, 'r', 'DisplayName', 'Predicción');
legend;
title('Predicción del Consumo de Energía');
xlabel('Tiempo');
ylabel('Consumo de Energía');
hold off;
