SF Fire Department Data Analysis
------------------
Data source: https://data.sfgov.org/api/views/nuek-vuh3
* operations.csv:
    - call_number: A unique 9-digit number assigned by the 911 Dispatch Center (DEM) to this call. These number are used for both Police and Fire calls.
    - unit_id: Unit Identifier. For example E01 for Engine 1 or T01 for Truck 1.
    - incident_number: A unique 8-digit number assigned by DEM to this Fire incident.
    - call_type: Type of call the incident falls into. See the list below.
    - call_date: Date the call is received at the 911 Dispatch Center. Used for reporting purposes.
    - watch_date: Watch date when the call is received. Watch date starts at 0800 each morning and ends at 0800 the next day.
    - received_timestamp: Date and time of call is received at the 911 Dispatch Center.
    - entry_timestamp: Date and time the 911 operator submits the entry of the initial call information into the CAD system
    - dispatch_timestamp: Date and time the 911 operator dispatches this unit to the call.
    - response_timestamp: Date and time this unit acknowledges the dispatch and records that the unit is en route to the location of the call.
    - on_scene_timestamp: Date and time the unit records arriving to the location of the incident
    - transport_timestamp: If this unit is an ambulance, date and time the unit begins the transport unit arrives to hospital
    - hospital_timestamp: If this unit is an ambulance, date and time the unit arrives to the hospital.
    - call_final_disposition: Disposition of the call (Code). For example TH2: Transport to Hospital - Code 2, FIR: Resolved by Fire Department
    - available_timestamp: Date and time this unit is not longer assigned to this call and it is available for another dispatch.
    - address: Address of midblock point associated with incident (obfuscated address to protect caller privacy)
    - city: City of incident
    - zipcode_of_incident: Zipcode of incident
    - battalion: Emergency Response District (There are 9 Fire Emergency Response Districts)
    - station_area: Fire Station First Response Area associated with the address of the incident
    - box: Fire box associated with the address of the incident. A box is the smallest area used to divide the City. Each box is associated with a unique unit dispatch order. (2400b)
    - original_priority: Initial call priority (Code 2: Non-Emergency or Code 3:Emergency).
    - priority: Call priority (Code 2: Non-Emergency or Code 3:Emergency).
    - final_priority: Final call priority (Code 2: Non-Emergency or Code 3:Emergency).
    - als_unit: Does this unit includes ALS (Advance Life Support) resources? Is there a paramedic in this unit?
    - call_type_group: Call types are divided into four main groups: Fire, Alarm, Potential Life Threatening and Non Life Threatening.
    - number_of_alarms: Number of alarms associated with the incident. This is a number between 1 and 5.
    - unit_type: Unit type
    - unit_sequence_in_call_dispatch: A number that indicates the order this unit was assigned to this call
    - fire_prevention_district: Bureau of Fire Prevention District associated with this address
    - supervisor_district: Supervisor District associated with this address
    - neighborhood_district: Neighborhood District associated with this address, boundaries available here: https://data.sfgov.org/d/p5b7-5n3h
    - location: Latitude and longitude of address obfuscated either to the midblock, intersection or call box
    - row_id: Unique identifier used for managing data updates. It is the concatenation of Call Number and Unit ID separated by a dash
    - latitude: Latitude of the address
    - longitude: Longitude of the address


To forecast
------------
* Already studied:
  - Response time

* Not yet studied:
  - Operations
  - Unit availability
  - Operation typology
  - Operation durations
  - Destination of the operations


Saturation Events
------------------
When a station has no more units available to dispatch, having the station full is not a problem.
The kind of unit, NORMAL or SPECIAL change the computation of the available units, specials are not
take in account.


Important Notes
----------------
Each station have an ENGINE assigned E1, E2, etc... From station No.1 to No.19 also a TRUCK with name T01, T02, etc...
Normally each unit is assigned to a specific station, ENGINE1 (E1) and TRUCK1 (T01) are assigned to STATION 1, in case
of necessity the units can be offered to other stations, for example TRUCK1 can help T02 and E2 of STATION 2.
This must be take in account during calculations of saturations problems. Look at EID: 100310192 to understand.

Some of the stations have also special units, these unit can lead to calculation errors because not all units can
partecipate to general operations; for example, a car incident can be solved by a fire boat. The first development
will not take care of specific units.


Internal studies
-----------------
https://sfbos.org/21-operations-division-reorganization


External Analysis
------------------
* https://www.arcgis.com/apps/MapSeries/index.html?appid=f98568a6ac70458dbccc60f407a23ac8
* https://www.arcgis.com/apps/MapJournal/index.html?appid=5df17d3178474b4f9d78b0f82f0d9789
* https://nbviewer.jupyter.org/github/yqzgh09/STA-141B/blob/master/final_project.html
* https://sfbos.org/management-audit-san-francisco-fire-department-summary
* https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8599738367597028/4332292154849829/3601578643761083/latest.html
* https://medium.com/wonks-this-way/san-francisco-emergencies-7d3facb64e1b
* https://medium.com/elitecommandtraining/fire-departments-are-response-models-not-production-models-f7943d5c623d
