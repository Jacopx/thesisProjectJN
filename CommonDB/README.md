CommonDB
-----------
The goal of this structure is to create a common place to store the data independently of the kind/source of it.
Having a common structure is useful to make the procedure of forecasting standard and get it always in the same way.
The previous version v1 had the problem to be too much generic, the 'Capacity Management Problem' (CMP) require a
more precise and fitted data model in order to correctly achieve the goal.

Structure
----------
Is composed of five tables:
* **event**: eid, start_dt, end_dt
* **involved**: eid, id
* **object**: id
* **info**: id, type, description
* **location**: id, latitude, longitude

The main table is EVENT, for each event there are one or more object involved and the table involved is used to link this relationship.
Info and Location are used to store more information about each object. Everything is enforced by some key=foreign_key pair.