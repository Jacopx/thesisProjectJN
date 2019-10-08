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

Utilization
------------
The process is based on the files contained in the folder 'data_to_load':
* ***data.csv***: Is the file with the data to be loaded in the system
* ***dict.txt***: [system_db:csv_field] Is a file of used to match the field of the DB with the field of the CSV

The usage of dict is upgraded respect v1.0, now also conditional mapping is available. 
For example: we need to map 3 different object, the start station, the ending station and the bike used.
Both stations require INFO and LOCATIONS entries, the bike is only an object. Mapping each feature allow the
creation of multiple tables for each object, adding only 2 of them, will prevent the creation of one table; set 
all the values to NONE will make an object without tables linked.

The system can also prevent to adding duplicate. For example: the destination of an operations is repeater for a 
number of times equals to the number of unit involved, setting the value to **ONE** force the system to add this entry
only one time, the unique name will be composed by a prefix, a dash and the eid string. In case of **UNIQUE**
the system will add more time this feature and make an unique key for them, this function is useful when is not possibile
to defined automatically an unique name.
