# Construcción features.

### Preguntas a responder:
1. ¿Como estan organizadas las columnas?
2. ¿Que familias de variables existen?
3. ¿Que ventanas temporales hay?
4. ¿Que columnas sirven para medir comportamiento?

# Feature Catalog

## Objetivo

Definir las features analíticas derivadas a partir de familias
de variables identificadas en las fuentes originales.

---

## Tabla: CANALES

### Familia: monto_*

Feature: ratio_digital_3m
Columnas base:
- monto_app_3m
- monto_web_3m
- monto_total_3m

Qué mide:
Nivel de adopción digital del cliente.

Hipótesis:
Clientes digitales tienen mayor probabilidad de débito recurrente.

---

### Familia: tx_*

Feature: intensidad_transaccional_3m

Qué mide:
Frecuencia de interacción financiera reciente.

---

## Tabla: MORAS

Feature: mora_promedio_6m

Qué mide:
Riesgo histórico del cliente.

---

## Tabla: GESTIONES

Feature: numero_gestiones_previas

Qué mide:
Dependencia de intervención humana.
