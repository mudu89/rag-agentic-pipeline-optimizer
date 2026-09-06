from snowflake.snowpark.functions import (
    col,
    upper,
    year
)

wo = session.table("UDX_CORE.UDX_MAXIMO.WORK_ORDER")
wl = session.table("UDX_CORE.UDX_MAXIMO.WORK_LOG")
person = session.table("UDX_CORE.UDX_MAXIMO.PERSON")

result = (
    wo
    .join(
        wl,
        wo["WORKORDERID"] == wl["WORKORDERID"]
    )
    .join(
        person,
        wo["OWNER"] == person["PERSONID"]
    )
    .filter(
        (upper(person["STATUS"]) == "ACTIVE") &
        (year(wo["REPORTDATE"]) == 2026)
    )
)

result.show()
