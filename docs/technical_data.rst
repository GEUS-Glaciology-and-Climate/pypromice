**********
Data types
**********

PROMICE automated weather station (AWS) data undergoes three main processing steps to reach a usable data standard. The output of each step forms an intermediary data version, which we label as sequential levels.


Level 0
=======
Level 0 is raw, untouched data in one of three formats:

- [X] Copied from CF card in the field (``raw``)
- [X] Downloaded from logger box in the field as a Slim Table Memory (``stm``)   
- [X] Transmitted via satellite and decoded (``tx``)


Level 1
=======
- [X] Engineering units (e.g. current or volts) converted to physical units (e.g. temperature or wind speed)
- [ ] Invalid/bad/suspicious data flagged
- [X] Multiple data files merged into one time series per station
  

Level 2
=======
- [X] Calibration using secondary sources (e.g. radiometric correction requires input of tilt sensor)
- [X] Observation corrections applied
- [X] Station position relative to sun calculated
  

Level 3
=======
- [X] Derived products calculated (e.g. sensible and latent heat flux)
- [X] Data merged, patched, and filled into one product
