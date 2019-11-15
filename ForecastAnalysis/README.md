Forecast Analysis
------------------
The purpose of this script is to perform analysis over the datasets loaded in CommonDB.


Functions
----------
* **Distance**: Calculate geodesic distance from source and destination point of events
* **Duration**: Calculate duration of events
* **Saturation**: Calculate amount of minute in saturation situations

Query
---------
```
SELECT date(date) as 'date', hour(date) as 'h', minute(date) as 'm', SUM(n) as 'variation'
FROM
     (
            SELECT *
            FROM        (SELECT start_dt as 'date', event.eid, 1 as n
                            FROM event, involved
                            WHERE event.dataset=involved.dataset AND event.eid=involved.eid and involved.type='dest' AND event.dataset='SFBS'
                            AND involved.id='70' AND etype<>'Saturation1' AND etype<>'Saturation2' AND etype<>'Saturation'
                            ORDER BY start_dt, eid
                        ) as coming
            UNION
            SELECT *
            FROM        (SELECT start_dt as 'date', event.eid, -1 as n
                            FROM event, involved
                            WHERE event.dataset=involved.dataset AND event.eid=involved.eid and involved.type='src' AND event.dataset='SFBS'
                            AND involved.id='70' AND etype<>'Saturation1' AND etype<>'Saturation2' AND etype<>'Saturation'
                            ORDER BY start_dt, eid
                        ) as outing
            ORDER BY date, eid
    ) as unione

GROUP BY date(date), hour(date), minute(date)
ORDER BY date(date), hour(date), minute(date);
```

```
SELECT date(end_dt), hour(end_dt), minute(end_dt), sum(n)
FROM (  SELECT *
        FROM (SELECT end_dt, 1 as n
                FROM (
                    SELECT src.eid, end_dt, src.id as 'src', dst.id as 'dst', lead(src.id) over (order by start_dt) as 'nextsrc'
                    FROM event, involved,
                         (  SELECT event.eid, involved.id
                            FROM event, involved
                            WHERE event.dataset=involved.dataset AND event.eid=involved.eid AND event.dataset='SFBS'
                             AND type='src'
                        ) as src,
                         (  SELECT event.eid, involved.id
                            FROM event, involved
                            WHERE event.dataset=involved.dataset AND event.eid=involved.eid AND event.dataset='SFBS'
                             AND type='dest'
                        ) as dst
                    WHERE event.dataset=involved.dataset AND event.eid=involved.eid  AND event.dataset='SFBS'
                         AND src.eid=event.eid AND dst.eid=event.eid AND type='with'
                    ORDER BY start_dt) as rides
            WHERE dst<>nextsrc AND nextsrc='70') as added
        UNION
        SELECT *
        FROM (SELECT end_dt, -1 as n
                FROM (
                    SELECT src.eid, end_dt, src.id as 'src', dst.id as 'dst', lead(src.id) over (order by start_dt) as 'nextsrc'
                    FROM event, involved,
                         (  SELECT event.eid, involved.id
                            FROM event, involved
                            WHERE event.dataset=involved.dataset AND event.eid=involved.eid AND event.dataset='SFBS'
                             AND type='src'
                        ) as src,
                         (  SELECT event.eid, involved.id
                            FROM event, involved
                            WHERE event.dataset=involved.dataset AND event.eid=involved.eid AND event.dataset='SFBS'
                             AND type='dest'
                        ) as dst
                    WHERE event.dataset=involved.dataset AND event.eid=involved.eid  AND event.dataset='SFBS'
                         AND src.eid=event.eid AND dst.eid=event.eid AND type='with'
                    ORDER BY start_dt) as rides
            WHERE dst<>nextsrc AND dst='70') as removed
        ORDER BY end_dt) as variation
GROUP BY date(end_dt), hour(end_dt), minute(end_dt)
ORDER BY date(end_dt), hour(end_dt), minute(end_dt);
```
