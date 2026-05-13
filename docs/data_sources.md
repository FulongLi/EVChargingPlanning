# Data Sources

The synthetic experiment uses London-like data generated from reproducible random seeds. The first real-data case uses London/Barnet open data through OpenStreetMap and Overpass.

Current London/Barnet real-data interface:

- OpenStreetMap `amenity=charging_station` for existing charging stations
- OpenStreetMap `amenity=parking` for candidate site proxies
- OpenStreetMap shops, offices, and selected amenities for commercial/activity demand
- OpenStreetMap major roads for accessibility proxies
- OpenStreetMap `power=substation` for grid-connection proxy construction
- GLA/London Datastore borough boundary metadata for study-area documentation

Future London/Barnet integration should target:

- public EV charging station locations
- population or household density
- points of interest
- road network or traffic-flow proxies
- parking or private driveway proxies
- borough boundaries
- grid connection or hosting-capacity proxies

Real distribution network topology is outside the MVP scope. The UK National Chargepoint Registry is not used as the default source because the GOV.UK guidance reports that it was decommissioned on 28 November 2024.
