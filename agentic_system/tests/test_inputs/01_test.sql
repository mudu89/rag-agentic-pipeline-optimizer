
/*
Rewrite as a JOIN with DISTINCT (or a semi-join, depending on the optimizer).
Ensure join on key columns.
Retrieve only necessary columns.
 */

SELECT
    WO.WORKORDERID,
    WO.DESCRIPTION
FROM UDX_CORE.UDX_MAXIMO.WORK_ORDER WO
WHERE EXISTS
(
    SELECT 1
    FROM UDX_CORE.UDX_MAXIMO.LABOR_TRANSACTION LT
    WHERE LT.WORKORDERID = WO.WORKORDERID
      AND LT.REGULARHRS > 8
);
