# Modelo de Datos Analítico
## Optimización de Cobranza en Mora Temprana

---

## 1. Unidad Analítica

La unidad analítica del modelo se define como:

cliente_id + obligacion_id + fecha_analisis

Esto permite analizar el comportamiento de pago a nivel obligación,
alineado con los procesos operativos de cobranza.

---

## 2. Arquitectura de Datos

El modelo sigue una arquitectura tipo Medallion:

### RAW Layer
Datos originales sin transformación.

Tablas:
- clientes_raw
- canales_raw
- moras_raw
- gestiones_raw
- excedentes_raw
- pagos_raw

---

### SILVER Layer
Datos integrados y normalizados.

Procesos:
- limpieza de datos
- estandarización de llaves
- joins cliente–obligación
- agregaciones temporales

Resultado:
dataset analítico consolidado.

---

### GOLD Layer
Dataset listo para modelado.

Tabla:
- dataset_modelo

Características:
- features comportamentales
- variables financieras
- historial transaccional
- variables de mora

---

## 3. Variable Objetivo del Modelo

Variable: var_rta

Definición:

var_rta = 1  
Cuando la obligación presenta pagos exclusivamente por débito recurrente
y recurrencia ≥ 40%.

var_rta = 0  
Cuando existen pagos por otros canales o combinaciones.

Tipo de problema:
Clasificación binaria supervisada.

Objetivo:
Estimar la probabilidad de pago mediante débito recurrente.

---

## 4. Salida del Modelo

El modelo genera:

prob_debito_recurrente  [0,1]

Interpretación:
Probabilidad estimada de pago exclusivamente por débito recurrente.

---

## 5. Regla Operativa de Negocio (Decision Layer)

La decisión operacional se define mediante reglas posteriores al modelo.

Variable derivada:
asignar_gestion

Definición:

asignar_gestion = 0 (No gestionar)
si:
    mora_actual entre 1 y 30 días
    AND prob_debito_recurrente ≥ threshold_operativo

asignar_gestion = 1 (Gestionar)
en otro caso.

---

## 6. Justificación de Negocio

El modelo permite:

- Reducir costos de gestión de cobranza.
- Priorizar recursos operativos.
- Identificar clientes con alta probabilidad de pago automático.

El valor de negocio se materializa mediante la definición
de un umbral operativo alineado con la estrategia de cobranza.

---

## 7. Métricas de Evaluación

Modelo:
- AUC (capacidad discriminativa)

Operación:
- tasa de clientes evitados en gestión
- riesgo operacional asociado al threshold

## 8. Diseño conceptual de Features.
Las variables utilizas por el modelo no existen directamente en las fuentes originales. 
Las features serán construidas mediante procesos de agregación, nomrmalización y calculo de metricas comportamentales a partir de datos historicos.

- Objetivo: Transformar eventos operativos en representaciones cuantificables del comportamiendo financiero del cliente.  

### Grupo 1 - Comportamiento de pago.
Hipótesis: Clientes consistentes tienden a automatizar pagos. 
Features:
- Frecuencia_pagos_3m
- ratio_pago_completo
- promedio_monto_pago
- variabilidad_pago
- porcentaje_excedentes

### Grupo 2 - Comportamiento de Mora.
Hipótesis: Clientes con mora baja estructural usan débito. 
Features:
- dias_mora_promedio_6m
- max_dias_mora
- numero_eventos_mora
- tendencia_mora

### Grupo 3 - Comportamiento de Uso de Canales.
Hipótesis: Usuarios digitales adoptan automatización. 
Features:
- procentaje_transacciones_app
- frecuencia_canal_digital
- diversidad_canales
- intendidad_transaccional

### Grupo 4 - Interacción de Cobranza.
Hipótesis: Clientes con baja gestión previa pagan solos. 
Features:
- numero_gestiones_previas
- tasa_acuerdos_pago
- rechazo_gestion_ratio

### Grupo 5 - Perfil Cliente.


## 9. Ventanas temporales.

Las variables se calcularán utilizando ventanas historicas:
- Ultimos 3 meses
- Ultimos 6 meses
- snapshot actual

## 10. Enfoque Analítico

Las features fueron diseñadas para capturar patrones comportamentales asociados al uso de debito recurrente, no unicamente estados financieros actuales. 

El objetivo es modelar el comportamiento futuro a partir de historia observable. 
