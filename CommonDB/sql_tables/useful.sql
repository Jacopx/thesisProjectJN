select event.eid, involved.type, info.type, info.id
from event, involved, info
where event.eid=involved.eid and event.dataset=involved.dataset
  and info.dataset=event.dataset and info.dataset=involved.dataset
  and info.id=involved.id and event.dataset='SFFD';

select i2.id as 'SS', i1.id as 'UNIT', count(*)
from involved as i1, involved as i2
where i2.type='start' and i1.type='act' and i1.dataset=i2.dataset and i1.dataset='SFFD' and i2.dataset='SFFD' and i1.eid=i2.eid
group by i2.id, i1.id
ORDER BY i2.id desc;

# SATURATION EVENTS FD
SELECT involved.id, event.eid, start_dt, end_dt, n.involved_unit, s.capacity
FROM event, involved, (select involved.dataset, eid, count(distinct info.id) as involved_unit
                            from involved, info
                            where info.dataset=involved.dataset and info.id=involved.id
                                and info.descr='N' and involved.type='act'
                            group by involved.dataset, eid) as n,
                        (select involved.dataset, eid, info.id, descr as capacity
                            from involved, info
                            where info.dataset=involved.dataset and info.id=involved.id
                                and involved.type='start'
                            ) as s
WHERE event.dataset=involved.dataset and event.eid=involved.eid
  and event.dataset='SFFD' and event.dataset=n.dataset and event.eid=n.eid and event.eid=s.eid
  and event.dataset=s.dataset and involved.id=s.id and n.dataset=s.dataset and n.eid=s.eid
ORDER BY id, start_dt;
