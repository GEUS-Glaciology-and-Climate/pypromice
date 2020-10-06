; IDL code by Dirk van As (GEUS) 2009-2018. Processes, corrects and averages raw automatic weather station data.

; Instructions:
; 1. Split data files at dates of maintenance visits so that sensor calibration coefficients are not applied to multiple sensors.
; 2. Replace all occurrences of "NAN", NAN, "INF", INF, -INF etc by -999 in the data files. In some cases (such as KAN_B) ',!M' needs to be removed if it is printed in a column of a used variable (tranmitted data only).
; 3. Place the data files in the "raw" station data folders.
; 4. Insert station and file metadata in the files named [AWS]_metadata.xlsx.
; 5. Enter station name(s) below and run.

; Suggested improvements to the code:
; eliminate the occasional NAN in SHF and LHF calculations
; make z0 site/season/surface-type-specific
; improve CloudCov calculation -> make site-specific or lat/elev dependent
; revisit tilt correction of SR or use the correction by Wang et al (TC, 2016)
; include netCDF outputing (wait for Charlie Zender's information on JAWS?)
; correct for shadow effects from the station frame itself 

;------------------------------------------------------------------------------------------------------------------------------

pro AWSdataprocessing_v3

; Automatic weather station name; also name of subdirectory (without 'raw')
;AWS = ['KPC_L','KPC_U','EGP','SCO_L','SCO_U','TAS_L','TAS_A','MIT','QAS_L','QAS_M','QAS_U','NUK_L','NUK_U','NUK_K','KAN_L','KAN_M','KAN_U','UPE_L','UPE_U','THU_L','THU_U','THU_U2','CEN'] ; run for current PROMICE stations
;AWS = ['KPC_L','KPC_U','SCO_L','SCO_U','TAS_L','TAS_A','QAS_L','QAS_M','QAS_U','NUK_L','NUK_U','NUK_K','UPE_L','UPE_U','THU_L','THU_U'] ; run for current PROMICE stations
;AWS = ['NUK_L','NUK_U'] ; run for current PROMICE stations
;AWS = ['KPC_L','KPC_U','SCO_L','SCO_U'] ; run for the old PROMICE stations
;AWS = ['THU_L','THU_U','THU_U2','TAS_L'];,'TAS_A']
;AWS = ['UPE_L','UPE_U','THU_L','THU_U','THU_U2','CEN'] ; run for a single PROMICE station
  AWS = ['EGP']                 ; run for a single PROMICE station

; dir = 'D:\CentOS\AWS_PROMICE\AWS_data_PROMICE\'
; datadir = 'data\'
  dir='./data/'
  datadir='./data/out/'
  version_no = '_v03'           ; Version of processing routine. Change when implementing a significant improvement / addition.
  columns_inst=53               ; Columns in inst. data file v03
  columns_hour=46               ; Columns in hourly data file v03
  columns_day =45               ; Columns in daily data file v03
  columns_mon =24               ; Columns in monthly data file v03

  set_tilt = 'no' ; If set to 'yes' the program will replace all tilt values by the values set below, which can be useful for stations with a malfunctioning inclinometer.
  set_tiltX = 0   ; KAN_U: 3.0, UPE_U: ~0
  set_tiltY = 0   ; KAN_U: -1.4, UPE_U: ~0

  testrun = 'no'                ; If set to 'yes' the program will only read and process the first 2 data files.
  updaterun = 'no' ; If set to 'yes' the program will only read and process the last number of "upd" data files. NB. still testing... RSF 2019, seems to work.
  upd=1            ; number of files to update from the bottom of the metadata file
  inst_output = 'yes' ; If 'yes', (large) data files with the same time stamps as the input data will be generated (“instantaneous” if 10-min logger data).

  for i_AWS=0,n_elements(AWS)-1 do begin
     metadatafile = 'metadata/'+AWS[i_AWS]+'_metadata.csv' ; Comma-separated file with AWS information (file names, calibration coefficients, lat/lon, etc.)
;metadatafile = 'metadata/'+AWS[i_AWS]+'_metadata1.csv' ; Comma-separated file with AWS information (file names, calibration coefficients, lat/lon, etc.)
     startmetadatafile=0
     nlines_meta=FILE_LINES(dir+metadatafile)
     if (nlines_meta lt upd) then updaterun='no'
;----------------------------------------------------------------------------------------------------
; Reading metadata and data -------------------------------------------------------------------------

     starttime = SYSTIME(1)     ; Used to determine how long the program takes to run

     header=''
     line_full=''
     year = -999
     month = -999
     day = -999
     hour = -999
     minute = -999
     month_cent = -999
     day_cent = -999
     day_year = -999
     p = -999
     T = -999
     Thc = -999
     RH = -999
     WS = -999
     WD = -999
     SRin = -999
     SRout = -999
     LRin = -999
     LRout = -999
     Trad = -999
     Haws = -999
     Haws_qual = -999
     Hstk = -999
     Hstk_qual = -999
     Hpt = -999
     Hpt_cor = -999
     Tice1 = -999
     Tice2 = -999
     Tice3 = -999
     Tice4 = -999
     Tice5 = -999
     Tice6 = -999
     Tice7 = -999
     Tice8 = -999
;ornt = -999
     tiltX = -999
     tiltY = -999
     tiltX_prev = -999
     tiltY_prev = -999
     GPStime = -999
     GPSlat = -999
     GPSlon = -999
     GPSelev = -999
     GPShdop = -999
     Tlog = -999
     Ifan = -999
     Vbat = -999
     datatype = 'x'
     lines_data_total = 0

     T_0 = 273.15

     if testrun eq 'yes' then numberofdatafiles = 2 else numberofdatafiles = nlines_meta-1
     if updaterun eq 'yes' then startmetadatafile = nlines_meta-1-upd else numberofdatafiles = nlines_meta-1 ; testing january 2019 RSF
     openr,unit,dir+metadatafile,/get_lun
     header = STRARR(startmetadatafile+1)
     readf,unit,header
     for j=startmetadatafile,numberofdatafiles-1 do begin ; will stop when metadata lines are all read

;  if eof(unit) then break
;  if updaterun eq 'yes' then readf,unit,line_full else readf,unit,line_full
        readf,unit,line_full
;  print,line_full

        line = strsplit(line_full,',',/extract)
        file_no = fix(line[0])
        filename = line[1]
        slimtablemem = line[2]
        transmitted = line[3]
        lines_hdr = fix(line[4])
        lines_data = long(line[5])
        lines_data_total = lines_data_total + lines_data
        columns_data = fix(line[6])
        year_start = fix(line[7])
        lat = float(line[8])
        lon = float(line[9])
        UTC_offset = fix(line[10])
        if UTC_offset ne float(line[10]) then print,'Alert! Only time offset in whole hours allowed in current program version!'
        Thc_offset = float(line[11]) ; Hygroclip temperature offset
        C_SRin = float(line[12])
        C_SRout = float(line[13])
        C_LRin = float(line[14])
        C_LRout = float(line[15])
        C_Hpt = float(line[16]) ; Pressure transducer calibration coefficient
        p_Hpt = float(line[17]) ; Air pressure at PT calibration
        f_Hpt = float(line[18]) ; This factor is 2.5 if the CR1000 data logger is used (1 for CR10X)
        af_Hpt = float(line[19]) ; Antifreeze percentage of solution in PT hose
;  rot_ini = float(line[20]) ; Doesn't correction station rotation yet - will be implemented later
        col_datetime = fix(line[21])
        col_date = fix(line[22])
        col_year = fix(line[23])
        col_month = fix(line[24])
        col_day = fix(line[25])
        col_day_year = fix(line[26])
        col_time = fix(line[27])
        col_hour = fix(line[28])
        col_min = fix(line[29])
        col_min_year = fix(line[30])
        col_p = fix(line[31])
        col_T = fix(line[32])
        col_Thc = fix(line[33])
        col_RH = fix(line[34])
        col_WS = fix(line[35])
        col_WD = fix(line[36])
        col_WDsd = fix(line[37])
        col_SRin = fix(line[38])
        col_SRout = fix(line[39])
        col_LRin = fix(line[40])
        col_LRout = fix(line[41])
        col_Trad = fix(line[42])
        col_Haws = fix(line[43])
        col_Haws_qual = fix(line[44])
        col_Hstk = fix(line[45])
        col_Hstk_qual = fix(line[46])
        col_Hpt = fix(line[47])
        col_Tice = fix(line[48:55])
        col_ornt = fix(line[56])
        col_tiltX = fix(line[57])
        col_tiltY = fix(line[58])
        col_GPStime = fix(line[59])
        col_GPSlat = fix(line[60])
        col_GPSlon = fix(line[61])
        col_GPSelev = fix(line[62])
        col_GPShdop = fix(line[66])
        col_Tlog = fix(line[67])
        col_Ifan = fix(line[68])
        col_Vbat_ini = fix(line[69])
        col_Vbat = fix(line[70])
        col_season = line[71]
        if col_Hpt ne 0 then begin
           if af_Hpt eq 50 then rho_af = 1092. else begin
              if af_Hpt eq 100 then rho_af = 1145. else begin
                 print,'Change program to allow for antifeeze mixtures other than 50% or 100%.'
                 break
              endelse
           endelse
        endif
;  print,filename
        openr,unit2,dir+'data_raw/'+AWS[i_AWS]+' raw/'+filename,/get_lun
        print,'File: ',filename
;  if lines_hdr gt 0 then for i=1,lines_hdr do readf,unit2,header
        if lines_hdr gt 0 then header = STRARR(lines_hdr) 
        if lines_hdr gt 0 then readf,unit2,header
        for i=0L,lines_data-1 do begin
           readf,unit2,line_full
           line = strsplit(line_full,',',/extract)
           datatype_temp = 'i'

           if col_datetime ne 0 then begin
              datetime = strsplit(line[col_datetime-1],'"- :',/extract)
              year_temp = fix(datetime[0])
              month_temp = fix(datetime[1])
              day_temp = fix(datetime[2])
              hour_temp = fix(datetime[3])
              minute_temp = fix(datetime[4])
              day_year_temp = julday(month_temp,day_temp,year_temp) - julday(1,1,year_temp) + 1
              day_cent_temp = julday(month_temp,day_temp,year_temp) - 2451544
              month_cent_temp = (year_temp-2000)*12 + month_temp
           endif
           if col_year ne 0 and col_day_year ne 0 and col_time ne 0 and col_datetime eq 0 then begin
              year_temp = fix(line[col_year-1])
              day_year_temp = fix(line[col_day_year-1])
              hour_temp = fix(line[col_time-1]/100)
              minute_temp = fix(line[col_time-1]) - 100*hour_temp
              day_cent_temp = day_year_temp + julday(1,1,year_temp) - julday(1,1,2000)
              caldat,day_cent_temp+2451544,month_temp,day_temp
              month_cent_temp = (year_temp-2000)*12 + month_temp
           endif
           if col_min_year ne 0 and col_datetime eq 0 and col_year eq 0 and col_day_year eq 0 and col_time eq 0 then begin ; if no date/time stamp in file, use minute in year to calculate date & time
              minute_year = long(line[col_min_year-1])
              if i eq 0 then year_temp = year_start
              if i ne 0 and minute_year eq 1440 then year_temp = year_temp+1
              if i ne 0 and minute_year gt 1440 and minute_year lt 2000 then print,'SEE IF YEAR TRANSITION IS OK'
              day_year_temp = (minute_year+1)/24/60 ; +1 minute is to avoid rounding errors
              day_cent_temp = day_year_temp + julday(1,1,year_temp) - 1 - 2451544
              caldat,day_cent_temp+2451544,month_temp,day_temp
              month_cent_temp = (year_temp-2000)*12+month_temp
              hour_temp = fix(24*((minute_year+1)/24./60.-day_year_temp)) ; +1 minute is to avoid rounding errors
              minute_temp = round(60*(minute_year/60.-minute_year/60))
           endif

           if slimtablemem eq 'yes' then begin ; change time stamp to start of the hour instead of end
              hour_temp = hour_temp - 1
              if hour_temp eq -1 then begin
                 hour_temp = 23
                 day_cent_temp = day_cent_temp - 1
                 caldat,day_cent_temp+2451544,month_temp,day_temp,year_temp
                 month_cent_temp = (year_temp-2000)*12+month_temp
                 day_year_temp = julday(month_temp,day_temp,year_temp) - julday(1,1,year_temp) + 1
              endif
              datatype_temp = 'h'
           endif
           if transmitted eq 'yes' then begin    ; change transmission time to start of the hour/day instead of end
              if col_season ne 0 then begin      ; if season identifier is transmitted
                 if line[col_season-1] eq '!W' then begin ; daily transmissions
                                ;day_cent_temp = day_cent_temp - 1
                    caldat,day_cent_temp+2451544,month_temp,day_temp,year_temp
                    month_cent_temp = (year_temp-2000)*12+month_temp
                    day_year_temp = julday(month_temp,day_temp,year_temp) - julday(1,1,year_temp) + 1
                    datatype_temp = 'd'
                 endif
                 if line[col_season-1] eq '!S' then begin ; hourly transmissions
                    hour_temp = hour_temp - 1
                    if hour_temp eq -1 then begin
                       hour_temp = 23
                                ;day_cent_temp = day_cent_temp - 1
                       caldat,day_cent_temp+2451544,month_temp,day_temp,year_temp
                       month_cent_temp = (year_temp-2000)*12+month_temp
                       day_year_temp = julday(month_temp,day_temp,year_temp) - julday(1,1,year_temp) + 1
                    endif
                    datatype_temp = 'h'
                 endif
              endif else begin                                     ; if season identifier is not transmitted
                 if day_year_temp lt 100 or day_year_temp gt 300 then begin ; daily transmissions
                                ;day_cent_temp = day_cent_temp - 1
                    caldat,day_cent_temp+2451544,month_temp,day_temp,year_temp
                    month_cent_temp = (year_temp-2000)*12+month_temp
                    day_year_temp = julday(month_temp,day_temp,year_temp) - julday(1,1,year_temp) + 1
                    datatype_temp = 'd'
                 endif else begin ; hourly transmissions
                    hour_temp = hour_temp - 1
                    if hour_temp eq -1 then begin
                       hour_temp = 23
                                ;day_cent_temp = day_cent_temp - 1
                       caldat,day_cent_temp+2451544,month_temp,day_temp,year_temp
                       month_cent_temp = (year_temp-2000)*12+month_temp
                       day_year_temp = julday(month_temp,day_temp,year_temp) - julday(1,1,year_temp) + 1
                    endif
                    datatype_temp = 'h'
                 endelse
              endelse
           endif
           if i eq 0 then print,'Date and time:',strcompress(year_temp)+'/'+strcompress(month_temp)+'/'+strcompress(day_temp),strcompress(hour_temp)+':'+strcompress(minute_temp)
           if UTC_offset ne 0 then begin
              hour_temp = hour_temp + UTC_offset
              if hour_temp lt 0 then begin
                 hour_temp = hour_temp + 24
                 day_cent_temp = day_cent_temp - 1
              endif
              if hour_temp gt 23 then begin
                 hour_temp = hour_temp - 24
                 day_cent_temp = day_cent_temp + 1
              endif
              caldat,day_cent_temp+2451544,month_temp,day_temp,year_temp
              month_cent_temp = (year_temp-2000)*12+month_temp
              day_year_temp = julday(month_temp,day_temp,year_temp) - julday(1,1,year_temp) + 1
              if i eq 0 then print,'Date and time after time correction: ',year_temp,month_temp,day_temp,hour_temp,minute_temp
           endif
           year = [year,year_temp]
           month = [month,month_temp]
           day = [day,day_temp]
           hour = [hour,hour_temp]
           minute = [minute,minute_temp]
           month_cent = [month_cent,month_cent_temp]
           day_cent = [day_cent,day_cent_temp]
           day_year = [day_year,day_year_temp]

           p = [p,float(line[col_p-1])]
           T = [T,float(line[col_T-1])]
           if col_Thc ne 0 then Thc = [Thc,float(line[col_Thc-1])-Thc_offset] else Thc = [Thc,-999] ; Removing HygroClip temperature offset
           RH = [RH,float(line[col_RH-1])]
;    RH = [RH,float(line[col_RH-1])-0.1*float(line[col_RH-1])]
           WS = [WS,float(line[col_WS-1])]
           WD = [WD,float(line[col_WD-1])]
           SRin = [SRin,float(line[col_SRin-1])*10/C_SRin] ; Calculating radiation (10^-5 V -> W/m2)
           SRout = [SRout,float(line[col_SRout-1])*10/C_SRout] ; Calculating radiation (10^-5 V -> W/m2)
           if float(line[col_LRin-1]) ne 0 then LRin = [LRin,float(line[col_LRin-1])*10/C_LRin + 5.67e-8*(float(line[col_Trad-1])+T_0)^4] else LRin = [LRin,-999] ; Calculating radiation (10^-5 V -> W/m2)
           if float(line[col_LRout-1]) ne 0 then LRout = [LRout,float(line[col_LRout-1])*10/C_LRout + 5.67e-8*(float(line[col_Trad-1])+T_0)^4] else LRout = [LRout,-999] ; Calculating radiation (10^-5 V -> W/m2)
           Trad = [Trad,float(line[col_Trad-1])]
           if col_Haws ne 0 and n_elements(line) ge col_Haws then Haws = [Haws,float(line[col_Haws-1])*((float(line[col_T-1])+T_0)/T_0)^0.5] else Haws = [Haws,-999] ; Correcting sonic ranger readings for sensitivity to air temperature
           if col_Haws_qual ne 0 and n_elements(line) ge col_Haws_qual then Haws_qual = [Haws_qual,float(line[col_Haws_qual-1])] else Haws_qual = [Haws_qual,-999]
           if col_Hstk ne 0 and n_elements(line) ge col_Hstk then Hstk = [Hstk,float(line[col_Hstk-1])*((float(line[col_T-1])+T_0)/T_0)^0.5] else Hstk = [Hstk,-999] ; Correcting sonic ranger readings for sensitivity to air temperature
           if col_Hstk_qual ne 0 and n_elements(line) ge col_Hstk_qual then Hstk_qual = [Hstk_qual,float(line[col_Hstk_qual-1])] else Hstk_qual = [Hstk_qual,-999]
           if col_Hpt ne 0 and n_elements(line) ge col_Hpt then Hpt = [Hpt,float(line[col_Hpt-1])*C_Hpt*F_Hpt*998./rho_af] else Hpt = [Hpt,-999] ; Calculating pressure transducer depth (V -> m water (rho=998 at calibration/room temperature) -> m antifreeze mix
           if col_Hpt ne 0 and n_elements(line) ge col_Hpt then Hpt_cor = [Hpt_cor,float(line[col_Hpt-1])*C_Hpt*F_Hpt*998./rho_af+100.*(p_Hpt-float(line[col_p-1]))/(rho_af*9.81)] else Hpt_cor = [Hpt_cor,-999] ; Calculating pressure transducer depth corrected f
           if col_Tice[0] ne 0 and n_elements(line) ge col_Tice[0] then Tice1 = [Tice1,float(line[col_Tice[0]-1])] else Tice1 = [Tice1,-999.9]
           if col_Tice[1] ne 0 and n_elements(line) ge col_Tice[1] then Tice2 = [Tice2,float(line[col_Tice[1]-1])] else Tice2 = [Tice2,-999]
           if col_Tice[2] ne 0 and n_elements(line) ge col_Tice[2] then Tice3 = [Tice3,float(line[col_Tice[2]-1])] else Tice3 = [Tice3,-999]
           if col_Tice[3] ne 0 and n_elements(line) ge col_Tice[3] then Tice4 = [Tice4,float(line[col_Tice[3]-1])] else Tice4 = [Tice4,-999]
           if col_Tice[4] ne 0 and n_elements(line) ge col_Tice[4] then Tice5 = [Tice5,float(line[col_Tice[4]-1])] else Tice5 = [Tice5,-999]
           if col_Tice[5] ne 0 and n_elements(line) ge col_Tice[5] then Tice6 = [Tice6,float(line[col_Tice[5]-1])] else Tice6 = [Tice6,-999]
           if col_Tice[6] ne 0 and n_elements(line) ge col_Tice[6] then Tice7 = [Tice7,float(line[col_Tice[6]-1])] else Tice7 = [Tice7,-999]
           if col_Tice[7] ne 0 and n_elements(line) ge col_Tice[7] then Tice8 = [Tice8,float(line[col_Tice[7]-1])] else Tice8 = [Tice8,-999]
;   if col_ornt ne 0 and n_elements(line) ge col_ornt then ornt = [ornt,float(line[col_ornt-1])] else ornt = [ornt,-999] ; NB: orientation of the AWS not used below
;    if col_tiltX ne 0 and n_elements(line) ge col_tiltX then begin
;      tiltX = [tiltX,float(line[col_tiltX-1])]
;      tiltX_prev = float(line[col_tiltX-1])
;      endif else tiltX = [tiltX,tiltX_prev]
;    if col_tiltY ne 0 and n_elements(line) ge col_tiltY then begin
;      tiltY = [tiltY,float(line[col_tilty-1])]
;      tiltY_prev = float(line[col_tilty-1])
;      endif else tiltY = [tiltY,tiltY_prev]
           if col_tiltX ne 0 and n_elements(line) ge col_tiltX then tiltX = [tiltX,float(line[col_tiltX-1])] else tiltX = [tiltX,-999]
           if col_tiltY ne 0 and n_elements(line) ge col_tiltY then tiltY = [tiltY,float(line[col_tilty-1])] else tiltY = [tiltY,-999]
           if col_GPStime ne 0 and n_elements(line) ge col_GPStime then GPStime_temp = long(strsplit(line[col_GPStime-1],'"GT',/extract)) else GPStime_temp = -999
           if col_GPSlat ne 0 and n_elements(line) ge col_GPSlat then GPSlat_temp = double(strsplit(line[col_GPSlat-1],'"NH',/extract)) else GPSlat_temp = -999
           if col_GPSlon ne 0 and n_elements(line) ge col_GPSlon then GPSlon_temp = double(strsplit(line[col_GPSlon-1],'"WH',/extract)) else GPSlon_temp = -999
           if col_GPSelev ne 0 and n_elements(line) ge col_GPSelev then GPSelev_temp = float(strsplit(line[col_GPSelev-1],'"',/extract)) else GPSelev_temp = -999
           if col_GPShdop ne 0 and n_elements(line) ge col_GPShdop then GPShdop_temp = float(strsplit(line[col_GPShdop-1],'"',/extract)) else GPShdop_temp = -999
           GPStime = [GPStime,GPStime_temp]
           if GPSlat_temp lt 90 and GPSlat_temp gt 0 then GPSlat_temp = GPSlat_temp + 100*fix(lat) ; Some stations only recorded minutes, not degrees
           if GPSlat_temp ne -999 then GPSlat_temp = fix(GPSlat_temp/100.)+(GPSlat_temp/100.-fix(GPSlat_temp/100.))*100./60.
           GPSlat = [GPSlat,GPSlat_temp]
           if GPSlon_temp lt 90 and GPSlon_temp gt 0 then GPSlon_temp = GPSlon_temp + 100*fix(lon) ; Some stations only recorded minutes, not degrees
           if GPSlon_temp ne -999 then GPSlon_temp = fix(GPSlon_temp/100.)+(GPSlon_temp/100.-fix(GPSlon_temp/100.))*100./60.
           GPSlon = [GPSlon,GPSlon_temp]
           GPSelev = [GPSelev,GPSelev_temp]
           GPShdop = [GPShdop,GPShdop_temp]
           if col_Tlog ne 0 and n_elements(line) ge col_Tlog then Tlog = [Tlog,float(line[col_Tlog-1])] else Tlog = [Tlog,-999]
           if col_Ifan ne 0 and n_elements(line) ge col_Ifan then Ifan = [Ifan,float(line[col_Ifan-1])] else Ifan = [Ifan,-999]
           if col_Vbat ne 0 and n_elements(line) ge col_Vbat then Vbat = [Vbat,float(line[col_Vbat-1])] else Vbat = [Vbat,-999]
           datatype = [datatype,datatype_temp]
        endfor
        free_lun,unit2
     endfor
     free_lun,unit

;----------------------------------------------------------------------------------------------------
; Recalculating tilt readings from voltage into degrees ---------------------------------------------
; (see shortwave radiation correction for more tilt calculations)

;tiltX = smooth(tiltX,7) & tiltY = smooth(tiltY,7)
; RSF - above line is not working for transmitted data 
     if transmitted ne 'yes' then begin
        tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
     endif
     notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
     tiltX = tiltX/10.
     tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
     if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 + 3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
     tiltY = tiltY/10.
     tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
     if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))

     if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.
     if n_elements(OKtiltY) gt 1 then tiltY[notOKtiltY] = interpol(tiltY[OKtiltY],OKtiltY,notOKtiltY) ; Interpolate over gaps for radiation correction; set to -999 again below.

     if set_tilt eq 'yes' then begin
        tiltX[*] = set_tiltX
        tiltY[*] = set_tiltY
        print,'NB: tilt values were overwritten by user-defined constants.'
     endif
;tilt_OK = where(tiltX gt -40 and tiltX lt 40 and tiltY gt -40 and tiltY lt 40)
;tiltX_rad[tilt_OK] = tiltX[tilt_OK]*!pi/180. ; degrees to radians
;tiltY_rad[tilt_OK] = tiltY[tilt_OK]*!pi/180. ; degrees to radians
     tiltX_rad = tiltX*!pi/180. ; degrees to radians
     tiltY_rad = tiltY*!pi/180. ; degrees to radians

;----------------------------------------------------------------------------------------------------
; Correcting relative humidity readings for T below 0 to give value with respect to ice -------------
     RH_cor = RH
     T_100 = T_0+100            ; steam point temperature in K
     ews = 1013.246             ; saturation pressure at steam point temperature, normal atmosphere
     ei0 = 6.1071
     e_s_wtr = 10.^(-7.90298*(T_100/(T+T_0)-1.) + 5.02808 * ALOG10(T_100/(T+T_0)) $ ; in hPa (Goff & Gratch)
                    - 1.3816E-7 * (10.^(11.344*(1.-(T+T_0)/T_100))-1.) + 8.1328E-3*(10.^(-3.49149*(T_100/(T+T_0)-1)) -1.) + ALOG10(ews))
     e_s_ice = 10.^(-9.09718 * (T_0 / (T+T_0) - 1.) - 3.56654 * ALOG10(T_0 / (T+T_0)) + 0.876793 * (1. - (T+T_0) / T_0) + ALOG10(ei0)) ; in hPa (Goff & Gratch)
     freezing = where(T lt 0 and T gt -100)
     if total(freezing) ne -1 then RH_cor[freezing] = RH[freezing]*e_s_wtr[freezing]/e_s_ice[freezing]

;----------------------------------------------------------------------------------------------------
; Calculating wind speed components and offset correction in wind direction when if sensor is not aligned north-south

; I'll worry about this once compasses are installed on the weather stations
;WD_offset = to be determined from compass
;WD = WD - WD_offset
;if WD_offset lt 0 then begin
;  outofrange = where(WD gt 360)
;  if total(outofrange) ne -1 then WD[outofrange] = WD[outofrange] - 360
;endif
;if WD_offset gt 0 then begin
;  outofrange = where(WD lt 0)
;  WD[outofrange] = WD[outofrange] + 360
;endif

     WSx = WS*sin(WD*!pi/180.)
     WSy = WS*cos(WD*!pi/180.)

;----------------------------------------------------------------------------------------------------
; Calculating cloud cover (for SRin correction) and surface temperature -----------------------------

     eps_overcast = 1.
     eps_clear = 9.36508e-6
     LR_overcast = eps_overcast*5.67e-8*(T+T_0)^4 ; assumption
     LR_clear = eps_clear*5.67e-8*(T+T_0)^6       ; Swinbank (1963)

;Special case for selected stations (will need this for all stations eventually)
     if AWS[i_AWS] eq 'KAN_M' then begin
        print,'KAN_M cloud cover calculations'
        LR_overcast = 315. + 4.*T
        LR_clear = 30 + 4.6e-13*(T+T_0)^6
     endif
     if AWS[i_AWS] eq 'KAN_U' then begin
        print,'KAN_U cloud cover calculations'
        LR_overcast = 305. + 4.*T
        LR_clear = 220 + 3.5*T
     endif

     CloudCov = (LRin-LR_clear)/(LR_overcast-LR_clear)
     overcast = where(CloudCov gt 1)
     clear = where(CloudCov lt 0)
     if total(overcast) ne -1 then CloudCov[overcast] = 1
     if total(clear) ne -1 then CloudCov[clear] = 0
     DifFrac = 0.2+0.8*CloudCov

     emissivity = 0.97
     Tsurf = ((LRout-(1-emissivity)*LRin)/emissivity/5.67e-8)^0.25 - T_0
     too_warm = where(Tsurf gt 0)
     if total(too_warm) ne -1 then Tsurf[too_warm] = 0

;----------------------------------------------------------------------------------------------------
; Correcting shortwave radiation --------------------------------------------------------------------

; Calculating tilt angle and direction of sensor and rotating to a north-south aligned coordinate system
     X = sin(tiltX_rad)*cos(tiltX_rad)*(sin(tiltY_rad))^2 + sin(tiltX_rad)*(cos(tiltY_rad))^2 ; Cartesian coordinate
     Y = sin(tiltY_rad)*cos(tiltY_rad)*(sin(tiltX_rad))^2 + sin(tiltY_rad)*(cos(tiltX_rad))^2 ; Cartesian coordinate
     Z = cos(tiltX_rad)*cos(tiltY_rad) + (sin(tiltX_rad))^2*(sin(tiltY_rad))^2                ; Cartesian coordinate
     phi_sensor_rad = -!pi/2.-atan(Y/X)                                                       ; spherical coordinate
     if total(where(X gt 0)) ne -1 then phi_sensor_rad[where(X gt 0)] = phi_sensor_rad[where(X gt 0)]+!pi
     if total(where(X eq 0 and Y lt 0)) ne -1 then phi_sensor_rad[where(X eq 0 and Y lt 0)] = !pi
     if total(where(X eq 0 and Y ge 0)) ne -1 then phi_sensor_rad[where(X eq 0 and Y ge 0)] = 0
     if total(where(phi_sensor_rad lt 0)) ne -1 then phi_sensor_rad[where(phi_sensor_rad lt 0)] = phi_sensor_rad[where(phi_sensor_rad lt 0)]+2*!pi
;phi_sensor_rad = phi_sensor_rad + phi_ini*!pi/180	; remove initial rotation of the station / sensor
     phi_sensor_deg = phi_sensor_rad*180./!pi ; radians to degrees
     theta_sensor_rad = acos(Z/(X^2+Y^2+Z^2)^0.5) ; spherical coordinate (or actually total tilt of the sensor, i.e. 0 when horizontal)
     theta_sensor_deg = theta_sensor_rad*180./!pi ; radians to degrees

; Offset correction (determine offset yourself using data for solar zenith angles larger than 110 deg)
; I actually don't do this as it shouldn't improve accuracy for well calibrated instruments
;SRin = SRin - SRin_offset
;SRout = SRout - SRout_offset

; Calculating zenith and hour angle of the sun
     d0_rad = 2.*!pi*(day_year+(hour+minute/60.)/24.-1.)/365.
     Declination_rad = asin(0.006918-0.399912*cos(d0_rad)+0.070257*sin(d0_rad)-0.006758*cos(2*d0_rad)+0.000907*sin(2*d0_rad)-0.002697*cos(3*d0_rad)+0.00148*sin(3*d0_rad))
     HourAngle_rad = 2.*!pi*(((hour+minute/60.)/24.-0.5) - lon/360.) ; - 15.*timezone/360.) ; NB: Make sure time is in UTC and longitude is positive when west! Hour angle should be 0 at noon.
     DirectionSun_deg = HourAngle_rad*180./!pi-180                   ; This is 180 deg at noon (NH), as opposed to HourAngle.
     if total(where(DirectionSun_deg lt 0)) ne -1 then DirectionSun_deg[where(DirectionSun_deg lt 0)] = DirectionSun_deg[where(DirectionSun_deg lt 0)]+360
     if total(where(DirectionSun_deg lt 0)) ne -1 then DirectionSun_deg[where(DirectionSun_deg lt 0)] = DirectionSun_deg[where(DirectionSun_deg lt 0)]+360
     ZenithAngle_rad = acos(cos(lat*!pi/180.)*cos(Declination_rad)*cos(HourAngle_rad) + sin(lat*!pi/180.)*sin(Declination_rad))
     ZenithAngle_deg = ZenithAngle_rad*180./!pi
     sundown = where(ZenithAngle_deg ge 90)
     SRtoa = 1372.*cos(ZenithAngle_rad) ; SRin at the top of the atmosphere
     if total(sundown) ne -1 then SRtoa[sundown] = 0

; Calculating the correction factor for direct beam radiation (http://solardat.uoregon.edu/SolarRadiationBasics.html)
     CorFac = sin(Declination_rad)*sin(lat*!pi/180.)*cos(theta_sensor_rad) $
              -sin(Declination_rad)*cos(lat*!pi/180.)*sin(theta_sensor_rad)*cos(phi_sensor_rad+!pi) $
              +cos(Declination_rad)*cos(lat*!pi/180.)*cos(theta_sensor_rad)*cos(HourAngle_rad) $
              +cos(Declination_rad)*sin(lat*!pi/180.)*sin(theta_sensor_rad)*cos(phi_sensor_rad+!pi)*cos(HourAngle_rad) $
              +cos(Declination_rad)*sin(theta_sensor_rad)*sin(phi_sensor_rad+!pi)*sin(HourAngle_rad)
     CorFac = cos(ZenithAngle_rad)/CorFac
     no_correction = where(CorFac le 0 or ZenithAngle_deg gt 90) ; sun out of field of view upper sensor
     if total(no_correction) ne -1 then CorFac[no_correction] = 1

; Calculating SRin over a horizontal surface corrected for station/sensor tilt
     CorFac_all = CorFac/(1.-DifFrac+CorFac*DifFrac)
     SRin_cor = SRin*CorFac_all

; Calculating albedo based on albedo values when sun is in sight of the upper sensor
     AngleDif_deg = 180./!pi*acos(sin(ZenithAngle_rad)*cos(HourAngle_rad+!pi)*sin(theta_sensor_rad)*cos(phi_sensor_rad)+sin(ZenithAngle_rad)*sin(HourAngle_rad+!pi)*sin(theta_sensor_rad)*sin(phi_sensor_rad)+cos(ZenithAngle_rad)*cos(theta_sensor_rad)) ; angle
;AngleDif_deg = 180./!pi*acos(cos(!pi/2.-ZenithAngle_rad)*cos(!pi/2.-theta_sensor_rad)*cos(HourAngle_rad-phi_sensor_rad)+sin(!pi/2.-ZenithAngle_rad)*sin(!pi/2.-theta_sensor_rad)) ; angle between sun and sensor
     albedo = SRout/SRin_cor
     OKalbedos = where(angleDif_deg lt 70 and ZenithAngle_deg lt 70 and albedo lt 1 and albedo gt 0, complement=notOKalbedos)
;OKalbedos = where(angleDif_deg lt 82.5 and ZenithAngle_deg lt 70 and albedo lt 1 and albedo gt 0, complement=notOKalbedos)
;The running mean calculation doesn't work for non-continuous data sets or variable temporal resolution (e.g. with multiple files)
;albedo_rm = 0*albedo
;albedo_rm[OKalbedos] = smooth(albedo[OKalbedos],obsrate+1,/edge_truncate) ; boxcar average of reliable albedo values
;albedo[notOKalbedos] = interpol(albedo_rm[OKalbedos],OKalbedos,notOKalbedos) ; interpolate over gaps
;albedo_rm[notOKalbedos] = albedo[notOKalbedos]
;So instead:
     albedo[notOKalbedos] = interpol(albedo[OKalbedos],OKalbedos,notOKalbedos) ; interpolate over gaps - gives problems for discontinuous data sets, but is not the end of the world

; Correcting SR using SRin when sun is in field of view of lower sensor assuming sensor measures only diffuse radiation
     sunonlowerdome = where(AngleDif_deg ge 90 and ZenithAngle_deg le 90)
     if total(sunonlowerdome) ne -1 then SRin_cor[sunonlowerdome] = SRin[sunonlowerdome]/DifFrac[sunonlowerdome]
     SRout_cor = SRout
     if total(sunonlowerdome) ne -1 then SRout_cor[sunonlowerdome] = albedo*SRin[sunonlowerdome]/DifFrac[sunonlowerdome]

; Setting SRin and SRout to zero for solar zenith angles larger than 95 deg or either SRin or SRout are (less than) zero
     no_SR = where(ZenithAngle_deg gt 95 or SRin_cor le 0 or SRout_cor le 0)
     if total(no_SR) ne -1 then begin
        SRin_cor[no_SR] = 0
        SRout_cor[no_SR] = 0
     endif

; Correcting SRin using more reliable SRout when sun not in sight of upper sensor
     SRin_cor = SRout_cor/albedo
     albedo[notOKalbedos] = -999
     albedo[OKalbedos[n_elements(OKalbedos)-1]:*] = -999 ; Removing albedos that were extrapolated (as opposed to interpolated) at the end of the time series - see above
     SRin_cor[OKalbedos[n_elements(OKalbedos)-1]:*] = -999 ; Removing the corresponding SRin_cor as well
     SRout_cor[OKalbedos[n_elements(OKalbedos)-1]:*] = -999 ; Removing the corresponding SRout_cor as well

; Removing spikes by interpolation based on a simple top-of-the-atmosphere limitation
     TOA_crit_nopass = where(SRin_cor gt 0.9*SRtoa+10)
     TOA_crit_pass = where(SRin_cor le 0.9*SRtoa+10)
     if total(TOA_crit_nopass) ne -1 then begin
        SRin_cor[TOA_crit_nopass] = interpol(SRin_cor[TOA_crit_pass],TOA_crit_pass,TOA_crit_nopass)
        SRout_cor[TOA_crit_nopass] = interpol(SRout_cor[TOA_crit_pass],TOA_crit_pass,TOA_crit_nopass)
     endif

     print,'- Sun in view of upper sensor / workable albedos:',n_elements(OKalbedos),100*n_elements(OKalbedos)/n_elements(SRin),'%'
     print,'- Sun below horizon:',n_elements(sundown),100*n_elements(sundown)/n_elements(SRin),'%'
     print,'- Sun in view of lower sensor:',n_elements(sunonlowerdome),100*n_elements(sunonlowerdome)/n_elements(SRin),'%'
     print,'- Spikes removed using TOA criteria:',n_elements(TOA_crit_nopass),100*n_elements(TOA_crit_nopass)/n_elements(SRin),'%'
     print,'- Mean net SR change by corrections:',total(SRin_cor-SRout_cor-SRin+SRout)/n_elements(SRin),' W/m2'

;----------------------------------------------------------------------------------------------------
; Removing values that fall outside of normal measurement range -------------------------------------

     p_bad = where(p lt 650 or p gt 1100)
     if total(p_bad) ne -1 then begin
        p[p_bad] = -999
        Hpt_cor[p_bad] = -999
     endif
     T_bad = where(T lt -80 or T gt 30)
     if total(T_bad) ne -1 then begin
        T[T_bad] = -999
        RH_cor[T_bad] = -999
        CloudCov[T_bad] = -999
        CorFac_all[T_bad] = -999
        SRin_cor[T_bad] = -999
        SRout_cor[T_bad] = -999
        Haws[T_bad] = -999
        Hstk[T_bad] = -999
     endif
     Thc_bad = where(Thc lt -80 or Thc gt 30 or Thc eq 0)
     if total(Thc_bad) ne -1 then Thc[Thc_bad] = -999
     RH_bad = where(RH le 0 or RH gt 150)
     if total(RH_bad) ne -1 then begin
        RH[RH_bad] = -999
        RH_cor[RH_bad] = -999
     endif
     RHcor_bad = where(RH_cor gt 100)
     if total(RHcor_bad) ne -1 then RH_cor[RHcor_bad] = 100
     WS_bad = where(WS lt 0 or WS gt 100)
     if total(WS_bad) ne -1 then begin
        WS[WS_bad] = -999
        WD[WS_bad] = -999
        WSx[WS_bad] = -999
        WSy[WS_bad] = -999
     endif
     WD_bad = where(WD lt 1 or WD gt 360)
     if total(WD_bad) ne -1 then begin
        WD[WD_bad] = -999
        WSx[WD_bad] = -999
        WSy[WD_bad] = -999
     endif
     SRin_bad = where(SRin lt -10 or SRin gt 1500)
     if total(SRin_bad) ne -1 then begin
        SRin[SRin_bad] = -999
        SRin_cor[SRin_bad] = -999
        SRout_cor[SRin_bad] = -999
        albedo[SRin_bad] = -999
     endif
     SRout_bad = where(SRout lt -10 or SRout gt 1000)
     if total(SRout_bad) ne -1 then begin
        SRout[SRout_bad] = -999
        SRin_cor[SRout_bad] = -999
        SRout_cor[SRout_bad] = -999
        albedo[SRout_bad] = -999
     endif
     LRin_bad = where(LRin lt 50 or LRin gt 500)
     if total(LRin_bad) ne -1 then begin
        LRin[LRin_bad] = -999
        Tsurf[LRin_bad] = -999
        CloudCov[LRin_bad] = -999
        CorFac_all[LRin_bad] = -999
        SRin_cor[LRin_bad] = -999
        SRout_cor[LRin_bad] = -999
        albedo[LRin_bad] = -999
     endif
     LRout_bad = where(LRout lt 50 or LRout gt 500)
     if total(LRout_bad) ne -1 then begin
        LRout[LRout_bad] = -999
        Tsurf[LRout_bad] = -999
     endif
     Trad_bad = where(Trad lt -80 or Trad gt 50)
     if total(Trad_bad) ne -1 then begin
        Trad[Trad_bad] = -999
        LRin[Trad_bad] = -999
        LRout[Trad_bad] = -999
        Tsurf[Trad_bad] = -999
     endif
     Haws_bad = where(Haws le 0.3 or Haws gt 3) ; SR50 doesn't give readings when H < ~0.35 m
     if total(Haws_bad) ne -1 then Haws[Haws_bad] = -999
     Haws_qual_bad = where(Haws_qual le 0)
     if total(Haws_qual_bad) ne -1 then Haws_qual[Haws_qual_bad] = -999
     Hstk_bad = where(Hstk le 0.3 or Hstk gt 8) ; SR50 doesn't give readings when H < ~0.35 m
     if total(Hstk_bad) ne -1 then Hstk[Hstk_bad] = -999
     Hstk_qual_bad = where(Hstk_qual le 0)
     if total(Hstk_qual_bad) ne -1 then Hstk_qual[Hstk_qual_bad] = -999
     Hpt_bad = where(Hpt le 0 or Hpt gt 30)
     if total(Hpt_bad) ne -1 then begin
        Hpt[Hpt_bad] = -999
        Hpt_cor[Hpt_bad] = -999
     endif
     Tice1_bad = where(Tice1 lt -80 or Tice1 gt 30)
     if total(Tice1_bad) ne -1 then Tice1[Tice1_bad] = -999
     Tice2_bad = where(Tice2 lt -80 or Tice2 gt 30)
     if total(Tice2_bad) ne -1 then Tice2[Tice2_bad] = -999
     Tice3_bad = where(Tice3 lt -80 or Tice3 gt 30)
     if total(Tice3_bad) ne -1 then Tice3[Tice3_bad] = -999
     Tice4_bad = where(Tice4 lt -80 or Tice4 gt 30)
     if total(Tice4_bad) ne -1 then Tice4[Tice4_bad] = -999
     Tice5_bad = where(Tice5 lt -80 or Tice5 gt 30)
     if total(Tice5_bad) ne -1 then Tice5[Tice5_bad] = -999
     Tice6_bad = where(Tice6 lt -80 or Tice6 gt 30)
     if total(Tice6_bad) ne -1 then Tice6[Tice6_bad] = -999
     Tice7_bad = where(Tice7 lt -80 or Tice7 gt 30)
     if total(Tice7_bad) ne -1 then Tice7[Tice7_bad] = -999
     Tice8_bad = where(Tice8 lt -80 or Tice8 gt 30)
     if total(Tice8_bad) ne -1 then Tice8[Tice8_bad] = -999
     tiltX_bad = where(tiltX lt -30 or tiltX gt 30)
     if total(tiltX_bad) ne -1 then begin
        tiltX[tiltX_bad] = -999
;  CorFac_all[tiltX_bad] = -999
        SRin_cor[tiltX_bad] = -999
        SRout_cor[tiltX_bad] = -999
        albedo[tiltX_bad] = -999
     endif
     tiltX[notOKtiltX] = -999
     tiltY_bad = where(tiltY lt -30 or tiltY gt 30)
     if total(tiltY_bad) ne -1 then begin
        tiltY[tiltY_bad] = -999
;  CorFac_all[tiltY_bad] = -999
        SRin_cor[tiltY_bad] = -999
        SRout_cor[tiltY_bad] = -999
        albedo[tiltY_bad] = -999
     endif
     tiltY[notOKtiltY] = -999
     GPStime_bad = where(GPStime le 0 or GPStime gt 240000)
     if total(GPStime_bad) ne -1 then GPStime[GPStime_bad] = -999
     GPSlat_bad = where(GPSlat lt 60 or GPSlat gt 83)
     if total(GPSlat_bad) ne -1 then GPSlat[GPSlat_bad] = -999
     GPSlon_bad = where(GPSlon lt 20 or GPSlon gt 70)
     if total(GPSlon_bad) ne -1 then GPSlon[GPSlon_bad] = -999
     GPSelev_bad = where(GPSelev le 0 or GPSelev gt 3000)
     if total(GPSelev_bad) ne -1 then GPSelev[GPSelev_bad] = -999
     GPShdop_bad = where(GPShdop le 0)
     if total(GPShdop_bad) ne -1 then GPShdop[GPShdop_bad] = -999
     Tlog_bad = where(Tlog lt -80 or Tlog gt 30)
     if total(Tlog_bad) ne -1 then Tlog[Tlog_bad] = -999
     Ifan_bad = where(Ifan lt -200 or Ifan gt 200)
     if total(Ifan_bad) ne -1 then Ifan[Ifan_bad] = -999
     Vbat_bad = where(Vbat le 0 or Vbat gt 30)
     if total(Vbat_bad) ne -1 then Vbat[Vbat_bad] = -999

;----------------------------------------------------------------------------------------------------
; Calculating hourly mean values ---------------------------------------------------------------------------
     hour_cent_h = (min(day_cent[where(day_cent gt 0)])-1)*24 + lindgen(24*(max(day_cent[where(day_cent gt 0)])-min(day_cent[where(day_cent gt 0)])+1)) ; no update version


     p_h = fltarr(n_elements(hour_cent_h))
     p_h[*] = -999
     T_h = p_h
     Thc_h = p_h
;RH_h = p_h
     RH_cor_h = p_h
;q_h = p_h
     WSx_h = p_h
     WSy_h = p_h
     WS_h = p_h
     WD_h = p_h
     SRin_h = p_h
     SRin_cor_h = p_h
     SRout_h = p_h
     SRout_cor_h = p_h
     albedo_h = p_h
     LRin_h = p_h
     LRout_h = p_h
     CloudCov_h = p_h
     Tsurf_h = p_h
     Haws_h = p_h
     Hstk_h = p_h
     Hpt_h = p_h
     Hpt_cor_h = p_h
     Tice1_h = p_h
     Tice2_h = p_h
     Tice3_h = p_h
     Tice4_h = p_h
     Tice5_h = p_h
     Tice6_h = p_h
     Tice7_h = p_h
     Tice8_h = p_h
     tiltX_h = p_h
     tiltY_h = p_h
     GPStime_h = p_h
     GPSlat_h = double(p_h)
     GPSlon_h = double(p_h)
     GPSelev_h = p_h
     GPShdop_h = p_h
     Tlog_h = p_h
     Ifan_h = p_h
     Vbat_h = p_h
     for i=0l,n_elements(hour_cent_h)-1 do begin
        sel = where((day_cent-1)*24+hour eq hour_cent_h[i] and datatype eq 'h') ;  first fill the arrays with available transmitted data
        if total(sel) ne -1 then begin
           p_h[i] = p[sel[0]]
           T_h[i] = T[sel[0]]
           Thc_h[i] = Thc[sel[0]]
;    RH_h[i] = RH[sel[0]]
           RH_cor_h[i] = RH_cor[sel[0]]
           WS_h[i] = WS[sel[0]]
           WSx_h[i] = WSx[sel[0]]
           WSy_h[i] = WSy[sel[0]]
           WD_h[i] = WD[sel[0]]
           SRin_h[i] = SRin[sel[0]]
           SRin_cor_h[i] = SRin_cor[sel[0]]
           SRout_h[i] = SRout[sel[0]]
           SRout_cor_h[i] = SRout_cor[sel[0]]
           albedo_h[i] = albedo[sel[0]]
           LRin_h[i] = LRin[sel[0]]
           LRout_h[i] = LRout[sel[0]]
           CloudCov_h[i] = CloudCov[sel[0]]
           Tsurf_h[i] = Tsurf[sel[0]]
           Haws_h[i] = Haws[sel[0]]
           Hstk_h[i] = Hstk[sel[0]]
           Hpt_h[i] = Hpt[sel[0]]
           Hpt_cor_h[i] = Hpt_cor[sel[0]]
           Tice1_h[i] = Tice1[sel[0]]
           Tice2_h[i] = Tice2[sel[0]]
           Tice3_h[i] = Tice3[sel[0]]
           Tice4_h[i] = Tice4[sel[0]]
           Tice5_h[i] = Tice5[sel[0]]
           Tice6_h[i] = Tice6[sel[0]]
           Tice7_h[i] = Tice7[sel[0]]
           Tice8_h[i] = Tice8[sel[0]]
           tiltX_h[i] = tiltX[sel[0]]
           tiltY_h[i] = tiltY[sel[0]]
           GPStime_h[i] = GPStime[sel[0]]
           GPSlat_h[i]  = GPSlat[sel[0]]
           GPSlon_h[i]  = GPSlon[sel[0]]
           GPSelev_h[i] = GPSelev[sel[0]]
           GPShdop_h[i] = GPShdop[sel[0]]
           Tlog_h[i] = Tlog[sel[0]]
           Ifan_h[i] = Ifan[sel[0]]
           Vbat_h[i] = Vbat[sel[0]]
        endif
        
        sel = where((day_cent-1)*24+hour eq hour_cent_h[i] and datatype eq 'i') ; this should overwrite the transmitted hourly values in case of overlap
        if total(sel) ne -1 then begin
           if total(where(p[sel] ne -999)) ne -1 then p_h[i] = mean(p[sel[where(p[sel] ne -999)]])
           if total(where(T[sel] ne -999)) ne -1 then T_h[i] = mean(T[sel[where(T[sel] ne -999)]])
           if total(where(Thc[sel] ne -999)) ne -1 then Thc_h[i] = mean(Thc[sel[where(Thc[sel] ne -999)]])
;    if total(where(RH[sel] ne -999)) ne -1 then RH_h[i] = mean(RH[sel[where(RH[sel] ne -999)]])
           if total(where(RH_cor[sel] ne -999)) ne -1 then RH_cor_h[i] = mean(RH_cor[sel[where(RH_cor[sel] ne -999)]])
           if total(where(WS[sel] ne -999)) ne -1 then WS_h[i] = mean(WS[sel[where(WS[sel] ne -999)]])
           if total(where(WSx[sel] ne -999)) ne -1 then WSx_h[i] = mean(WSx[sel[where(WSx[sel] ne -999)]])
           if total(where(WSy[sel] ne -999)) ne -1 then WSy_h[i] = mean(WSy[sel[where(WSy[sel] ne -999)]])
           if total(where(SRin[sel] ne -999)) ne -1 then SRin_h[i] = mean(SRin[sel[where(SRin[sel] ne -999)]])
           if total(where(SRin_cor[sel] ne -999)) ne -1 then SRin_cor_h[i] = mean(SRin_cor[sel[where(SRin_cor[sel] ne -999)]])
           if total(where(SRout[sel] ne -999)) ne -1 then SRout_h[i] = mean(SRout[sel[where(SRout[sel] ne -999)]])
           if total(where(SRout_cor[sel] ne -999)) ne -1 then SRout_cor_h[i] = mean(SRout_cor[sel[where(SRout_cor[sel] ne -999)]])
           if total(where(albedo[sel] ne -999)) ne -1 then albedo_h[i] = mean(albedo[sel[where(albedo[sel] ne -999)]])
           if total(where(LRin[sel] ne -999)) ne -1 then LRin_h[i] = mean(LRin[sel[where(LRin[sel] ne -999)]])
           if total(where(LRout[sel] ne -999)) ne -1 then LRout_h[i] = mean(LRout[sel[where(LRout[sel] ne -999)]])
           if total(where(CloudCov[sel] ne -999)) ne -1 then CloudCov_h[i] = mean(CloudCov[sel[where(CloudCov[sel] ne -999)]])
           if total(where(Tsurf[sel] ne -999)) ne -1 then Tsurf_h[i] = mean(Tsurf[sel[where(Tsurf[sel] ne -999)]])
           if total(where(Haws[sel] ne -999)) ne -1 then Haws_h[i] = mean(Haws[sel[where(Haws[sel] ne -999)]])
           if total(where(Hstk[sel] ne -999)) ne -1 then Hstk_h[i] = mean(Hstk[sel[where(Hstk[sel] ne -999)]])
           if total(where(Hpt[sel] ne -999)) ne -1 then Hpt_h[i] = mean(Hpt[sel[where(Hpt[sel] ne -999)]])
           if total(where(Hpt_cor[sel] ne -999)) ne -1 then Hpt_cor_h[i] = mean(Hpt_cor[sel[where(Hpt_cor[sel] ne -999)]])
           if total(where(Tice1[sel] ne -999)) ne -1 then Tice1_h[i] = mean(Tice1[sel[where(Tice1[sel] ne -999)]])
           if total(where(Tice2[sel] ne -999)) ne -1 then Tice2_h[i] = mean(Tice2[sel[where(Tice2[sel] ne -999)]])
           if total(where(Tice3[sel] ne -999)) ne -1 then Tice3_h[i] = mean(Tice3[sel[where(Tice3[sel] ne -999)]])
           if total(where(Tice4[sel] ne -999)) ne -1 then Tice4_h[i] = mean(Tice4[sel[where(Tice4[sel] ne -999)]])
           if total(where(Tice5[sel] ne -999)) ne -1 then Tice5_h[i] = mean(Tice5[sel[where(Tice5[sel] ne -999)]])
           if total(where(Tice6[sel] ne -999)) ne -1 then Tice6_h[i] = mean(Tice6[sel[where(Tice6[sel] ne -999)]])
           if total(where(Tice7[sel] ne -999)) ne -1 then Tice7_h[i] = mean(Tice7[sel[where(Tice7[sel] ne -999)]])
           if total(where(Tice8[sel] ne -999)) ne -1 then Tice8_h[i] = mean(Tice8[sel[where(Tice8[sel] ne -999)]])
           if total(where(tiltX[sel] ne -999)) ne -1 then tiltX_h[i] = mean(tiltX[sel[where(tiltX[sel] ne -999)]])
           if total(where(tiltY[sel] ne -999)) ne -1 then tiltY_h[i] = mean(tiltY[sel[where(tiltY[sel] ne -999)]])
           GPStime_h[i] = GPStime[sel[0]]
           GPSlat_h[i]  = GPSlat[sel[0]]
           GPSlon_h[i]  = GPSlon[sel[0]]
           GPSelev_h[i] = GPSelev[sel[0]]
           GPShdop_h[i] = GPShdop[sel[0]]
           if total(where(Tlog[sel] ne -999)) ne -1 then Tlog_h[i] = mean(Tlog[sel[where(Tlog[sel] ne -999)]])
           if total(where(Ifan[sel] ne -999)) ne -1 then Ifan_h[i] = mean(Ifan[sel[where(Ifan[sel] ne -999)]])
           if total(where(Vbat[sel] ne -999)) ne -1 then Vbat_h[i] = mean(Vbat[sel[where(Vbat[sel] ne -999)]])
;    OK = where(Haws_h[i] ne -999)
;    if n_elements(OK) ne -1 then Haws_h = interpolate(Haws_h[OK],Haws_h[i], MISSING=-999) else begin
;     T_fit = [-999,-999] & fit_stdev = [-999,-999]
;    endif
        endif
     endfor
     if AWS[i_AWS] eq 'CEN' then begin  
        bad = Where(Haws_h eq -999, nbad, COMPLEMENT=good, NCOMPLEMENT=ngood)
        IF nbad GT 0 && ngood GT 1 THEN Haws_h[bad] = INTERPOL(Haws_h[good], good, bad)
        PRINT, 'interpolate ', n_elements(Haws_h)
     endif
     day_cent_h = hour_cent_h/24+1
     hour_h = hour_cent_h - (day_cent_h-1)*24
     caldat,day_cent_h+2451544,month_h,day_h,year_h
     day_year_h = JulDay(month_h,day_h,year_h)-JulDay(12,31,year_h-1)
     if total(where(WSx_h ne -999)) ne -1 then WD_h[where(WSx_h ne -999)] = 180./!pi*atan(WSx_h[where(WSx_h ne -999)]/WSy_h[where(WSx_h ne -999)])
     if total(where(WSy_h lt 0 and WSy_h ne -999)) ne -1 then WD_h[where(WSy_h lt 0 and WSy_h ne -999)] = WD_h[where(WSy_h lt 0 and WSy_h ne -999)]-180
     if total(where(WD_h lt 0 and WD_h ne -999)) ne -1 then WD_h[where(WD_h lt 0 and WD_h ne -999)] = WD_h[where(WD_h lt 0 and WD_h ne -999)]+360
     if total(where(WSy_h eq 0 and WSx_h gt 0)) ne -1 then WD_h[where(WSy_h eq 0 and WSx_h gt 0)] = 90
     if total(where(WSy_h eq 0 and WSx_h lt 0)) ne -1 then WD_h[where(WSy_h eq 0 and WSx_h lt 0)] = 270

     Hpt_cor_h_smooth = smooth(Hpt_cor_h,5)

;----------------------------------------------------------------------------------------------------
; Calculating turbulent heat fluxes -----------------------------------------------------------------
; NB: Requires (hourly) averages. Only variables feeding into and out of this section have subscript "h", but all are hourly values.
;q_h = RH_cor_h
; Constant declariation
     z_0    =    0.001          ; aerodynamic surface roughness length for momention (assumed constant for all ice/snow surfaces)
     eps    =    0.622
     es_0   =    6.1071         ; saturation vapour pressure at the melting point (hPa)
     es_100 = 1013.246          ; saturation vapour pressure at steam point temperature (hPa)
     g      =    9.82           ; gravitational acceleration (m/s2)
     gamma  =   16.             ; flux profile correction (Paulson & Dyer)
     kappa  =    0.4            ; Von Karman constant (0.35-0.42)
     L_sub  =    2.83e6         ; latent heat of sublimation (J/kg)
     R_d    =  287.05           ; gas constant of dry air
     aa     =    0.7            ; flux profile correction constants (Holtslag & De Bruin '88)
     bb     =    0.75
     cc     =    5.
     dd     =    0.35
     c_pd   = 1005.             ; specific heat of dry air (J/kg/K)
     WS_lim =    1.
     L_dif_max = 0.01

; Array declaration and initial guesses
     z_WS = Haws_h + 0.4
     z_T = Haws_h - 0.1
     rho_atm = 100.*p_h/R_d/(T_h+T_0)                             ; atmospheric density
     mu = 18.27e-6*(291.15+120)/((T_h+T_0)+120)*((T_h+T_0)/291.15)^1.5 ; dynamic viscosity of air (Pa s) (Sutherlands' equation using C = 120 K)
     nu = mu/rho_atm                                                   ; kinematic viscosity of air (m^2/s)
     u_star = kappa*WS_h/alog(z_WS/z_0)
     Re = u_star*z_0/nu
     z_0h = z_0*exp(1.5-0.2*alog(Re)-0.11*(alog(Re))^2) ; rough surfaces: Smeets & Van den Broeke 2008
     z_0h[where(WS_h le 0)] = 1e-10
     es_ice_surf = 10.^(-9.09718*(T_0/(Tsurf_h+T_0)-1.) - 3.56654*ALOG10(T_0/(Tsurf_h+T_0))+0.876793*(1.-(Tsurf_h+T_0)/T_0) + ALOG10(es_0))
     q_surf = eps*es_ice_surf/(p_h-(1-eps)*es_ice_surf)
     es_wtr = 10.^(-7.90298*(T_100/(T_h+T_0)-1.) + 5.02808 * ALOG10(T_100/(T_h+T_0)) $ ; saturation vapour pressure above 0 C (hPa)
                   - 1.3816E-7 * (10.^(11.344*(1.-(T_h+T_0)/T_100))-1.) $
                   + 8.1328E-3*(10.^(-3.49149*(T_100/(T_h+T_0)-1)) -1.) + ALOG10(es_100))
     es_ice = 10.^(-9.09718 * (T_0 / (T_h+T_0) - 1.) - 3.56654 * ALOG10(T_0 / (T_h+T_0)) + 0.876793 * (1. - (T_h+T_0) / T_0) + ALOG10(es_0)) ; saturation vapour pressure below 0 C (hPa)
     q_sat = eps * es_wtr/(p_h-(1-eps)*es_wtr) ; specific humidity at saturation (incorrect below melting point)
     freezing = where(T_h lt 0)                ; replacing saturation specific humidity values below melting point
     q_sat[freezing] = eps * es_ice[freezing]/(p_h[freezing]-(1-eps)*es_ice[freezing])
     q_h = RH_cor_h*q_sat/100   ; specific humidity in kg/kg
     theta = T_h + z_T*g/c_pd
     SHF_h = T_h & SHF_h[*] = 0 & LHF_h = SHF_h & L = SHF_h+1e5

     stable   = where(theta ge Tsurf_h and WS_h gt WS_lim and T_h ne -999 and Tsurf_h ne -999 and RH_cor_h ne -999 and p_h ne -999 and Haws_h ne -999)
     unstable = where(theta lt Tsurf_h and WS_h gt WS_lim and T_h ne -999 and Tsurf_h ne -999 and RH_cor_h ne -999 and p_h ne -999 and Haws_h ne -999)
;no_wind  = where( WS_h ne -999    and WS_h le WS_lim and T_h ne -999 and Tsurf_h ne -999 and RH_cor_h ne -999 and p_h ne -999 and Haws_h ne -999)

     for i=0,30 do begin        ; stable stratification
        psi_m1 = -(aa*         z_0/L[stable] + bb*(         z_0/L[stable]-cc/dd)*exp(-dd*         z_0/L[stable]) + bb*cc/dd)
        psi_m2 = -(aa*z_WS[stable]/L[stable] + bb*(z_WS[stable]/L[stable]-cc/dd)*exp(-dd*z_WS[stable]/L[stable]) + bb*cc/dd)
        psi_h1 = -(aa*z_0h[stable]/L[stable] + bb*(z_0h[stable]/L[stable]-cc/dd)*exp(-dd*z_0h[stable]/L[stable]) + bb*cc/dd)
        psi_h2 = -(aa* z_T[stable]/L[stable] + bb*( z_T[stable]/L[stable]-cc/dd)*exp(-dd* z_T[stable]/L[stable]) + bb*cc/dd)
        u_star[stable] = kappa*WS_h[stable]/(alog(z_WS[stable]/z_0)-psi_m2+psi_m1)
        Re[stable] = u_star[stable]*z_0/nu[stable]
        z_0h[stable] = z_0*exp(1.5-0.2*alog(Re[stable])-0.11*(alog(Re[stable]))^2)
        if n_elements(where(z_0h[stable] lt 1e-6)) gt 1 then z_0h[stable[where(z_0h[stable] lt 1e-6)]] = 1e-6
        th_star = kappa*(theta[stable]-Tsurf_h[stable])/(alog(z_T[stable]/z_0h[stable])-psi_h2+psi_h1)
        q_star  = kappa*(  q_h[stable]- q_surf[stable])/(alog(z_T[stable]/z_0h[stable])-psi_h2+psi_h1)
        SHF_h[stable] = rho_atm[stable]*c_pd *u_star[stable]*th_star
        LHF_h[stable] = rho_atm[stable]*L_sub*u_star[stable]* q_star
        L_prev = L[stable]
        L[stable] = u_star[stable]^2*(theta[stable]+T_0)*(1+((1-eps)/eps)*q_h[stable])/(g*kappa*th_star*(1+((1-eps)/eps)*q_star))
        L_dif = abs((L_prev-L[stable])/L_prev)
;  print,"HF iterations stable stratification: ",i+1,n_elements(where(L_dif gt L_dif_max)),100.*n_elements(where(L_dif gt L_dif_max))/n_elements(where(L_dif))
        if n_elements(where(L_dif gt L_dif_max)) eq 1 then break
     endfor

     if n_elements(unstable) gt 1 then begin
        for i=0,20 do begin     ; unstable stratification
           x1  = (1-gamma*z_0           /L[unstable])^0.25
           x2  = (1-gamma*z_WS[unstable]/L[unstable])^0.25
           y1  = (1-gamma*z_0h[unstable]/L[unstable])^0.5
           y2  = (1-gamma*z_T[unstable] /L[unstable])^0.5
           psi_m1 = alog(((1+x1)/2)^2*(1+x1^2)/2)-2*atan(x1)+!pi/2
           psi_m2 = alog(((1+x2)/2)^2*(1+x2^2)/2)-2*atan(x2)+!pi/2
           psi_h1 = alog(((1+y1)/2)^2)
           psi_h2 = alog(((1+y2)/2)^2)
           u_star[unstable] = kappa*WS_h[unstable]/(alog(z_WS[unstable]/z_0)-psi_m2+psi_m1)
           Re[unstable] = u_star[unstable]*z_0/nu[unstable]
           z_0h[unstable] = z_0*exp(1.5-0.2*alog(Re[unstable])-0.11*(alog(Re[unstable]))^2)
           if n_elements(where(z_0h[unstable] lt 1e-6)) gt 1 then z_0h[unstable[where(z_0h[unstable] lt 1e-6)]] = 1e-6
           th_star = kappa*(theta[unstable]-Tsurf_h[unstable])/(alog(z_T[unstable]/z_0h[unstable])-psi_h2+psi_h1)
           q_star  = kappa*(  q_h[unstable]- q_surf[unstable])/(alog(z_T[unstable]/z_0h[unstable])-psi_h2+psi_h1)
           SHF_h[unstable] = rho_atm[unstable]*c_pd *u_star[unstable]*th_star
           LHF_h[unstable] = rho_atm[unstable]*L_sub*u_star[unstable]* q_star
           L_prev = L[unstable]
           L[unstable] = u_star[unstable]^2*(theta[unstable]+T_0)*(1+((1-eps)/eps)*q_h[unstable])/(g*kappa*th_star*(1+((1-eps)/eps)*q_star))
           L_dif = abs((L_prev-L[unstable])/L_prev)
;    print,"HF iterations unstable stratification: ",i+1,n_elements(where(L_dif gt L_dif_max)),100.*n_elements(where(L_dif gt L_dif_max))/n_elements(where(L_dif))
           if n_elements(where(L_dif gt L_dif_max)) eq 1 then break
        endfor
     endif

     q_h = 1000.*q_h            ; from kg/kg to g/kg
;no_q = where(p_h eq -999 or T_h eq -999 or RH_cor_h eq -999)
;if total(no_q) ne -1 then q_h[no_q] = -999
;no_HF  = where(p_h eq -999 or T_h eq -999 or Tsurf_h eq -999 or WS_h eq -999 or Haws_h eq -999)
     no_HF = where(p_h eq -999 or T_h eq -999 or Tsurf_h eq -999 or RH_cor_h eq -999 or WS_h eq -999 or Haws_h eq -999)
     no_qh = where(T_h eq -999 or RH_cor_h eq -999 or p_h eq -999 or Tsurf_h eq -999)
;no_SHF = where(p_h eq -999 or T_h eq -999 or Tsurf_h eq -999 or WS_h eq -999 or Haws_h eq -999)
;no_LHF = where(p_h eq -999 or T_h eq -999 or Tsurf_h eq -999 or RH_cor_h eq -999 or WS_h eq -999 or Haws_h eq -999)
;if total(no_LHF) ne -1 then LHF_h[no_LHF] = -999
;if total(no_SHF) ne -1 then SHF_h[no_SHF] = -999
     if total(no_HF) ne -1 then begin
        SHF_h[no_HF] = -999
        LHF_h[no_HF] = -999
     endif
     if total(no_qh) ne -1 then begin
        q_h[no_qh] = -999
     endif
;print,'size q_h: ',size(q_h)

;----------------------------------------------------------------------------------------------------
; Calculating daily and monthly mean values ---------------------------------------------------------------------------

     MinCov = 0.8 ; minimum data coverage (0-1) for calculation of daily averages (NB: for variables that show little change over the course of one day only one measurement is enough)
     MinHDOP = 1.0              ; minimum HDOP value for calculation of daily lat, lon, and elevation averages
     MinDays = 24               ; minimum amount of days per month needed to calculate montly averages

     day_cent_d = min(day_cent[where(day_cent gt 0)]) + indgen(max(day_cent[where(day_cent gt 0)])-min(day_cent[where(day_cent gt 0)])+1)
     month_cent_m = min(month_cent[where(month_cent gt 0)]) + indgen(max(month_cent[where(month_cent gt 0)])-min(month_cent[where(month_cent gt 0)])+1)

     if updaterun eq 'yes' then begin
        a_t=0
        columns1 = columns_hour ;46 ;hour
        filename_old = dir+datadir+AWS[i_AWS]+'_hour'+version_no+'.txt'
        nlines_old=FILE_LINES(filename_old)
        line_old = fltarr(columns1,nlines_old-1)
        OPENR, lun, filename_old, /GET_LUN
        header = STRARR(1)
        READF, lun, header      ; Read one line at a time, saving the result into array
        READF, lun, line_old    ;_full
        CLOSE, lun
        free_lun,lun
        cen_old=line_old(5,0)
        hour_s=hour[1,0]
        index_h=24-hour_s+1
        cen_new=(min(day_cent[where(day_cent gt 0)])) ;*24;line_upd(5,0)
        FOR i=0,nlines_old-2 DO BEGIN
           IF (fix(line_old(5,i)) eq fix(cen_new) and fix(line_old(3,i)) eq fix(hour_s) ) THEN a_t=i ; and fix(line_old(3,i)) eq fix(cen_new)
        ENDFOR
        line_dum=fltarr(columns1,a_t)
        line_dum=line_old[*,0:a_t]
        hour_cent_h_min=24*(fix(cen_old))
        hour_cent_h_dif=24*(max(day_cent[where(day_cent gt 0)])-min(day_cent[where(day_cent gt 0)])+1)
        hour_cent_h = hour_cent_h_min + lindgen(hour_cent_h_dif)
        year_h      = [reform(line_dum[0,*]),year_h[index_h:*]]
        month_h     = [reform(line_dum[1,*]),month_h[index_h:*]]
        day_h       = [reform(line_dum[2,*]),day_h[index_h:*]]
        hour_h      = [reform(line_dum[3,*]),hour_h[index_h:*]]
        day_year_h  = [reform(line_dum[4,*]),day_year_h[index_h:*]]
        day_cent_h  = [reform(line_dum[5,*]),day_cent_h[index_h:*]]
        p_h         = [reform(line_dum[6,*]),p_h[index_h:*]]
        T_h         = [reform(line_dum[7,*]),T_h[index_h:*]]
        Thc_h       = [reform(line_dum[8,*]),Thc_h[index_h:*]]
                                ;RH_h        = [reform(line_dum[9,*]),RH_h[index_h:*]]
        RH_cor_h    = [reform(line_dum[9,*]),RH_cor_h[index_h:*]]
        q_h         = [reform(line_dum[10,*]),q_h[index_h:*]]
        WS_h        = [reform(line_dum[11,*]),WS_h[index_h:*]]
        WD_h        = [reform(line_dum[12,*]),WD_h[index_h:*]]
        SHF_h       = [reform(line_dum[13,*]),SHF_h[index_h:*]]
        LHF_h       = [reform(line_dum[14,*]),LHF_h[index_h:*]]
        SRin_h      = [reform(line_dum[15,*]),SRin_h[index_h:*]]
        SRin_cor_h  = [reform(line_dum[16,*]),SRin_cor_h[index_h:*]]
        SRout_h     = [reform(line_dum[17,*]),SRout_h[index_h:*]]
        SRout_cor_h = [reform(line_dum[18,*]),SRout_cor_h[index_h:*]]
        albedo_h    = [reform(line_dum[19,*]),albedo_h[index_h:*]]
        LRin_h      = [reform(line_dum[20,*]),LRin_h[index_h:*]]
        LRout_h     = [reform(line_dum[21,*]),LRout_h[index_h:*]]
        CloudCov_h  = [reform(line_dum[22,*]),CloudCov_h[index_h:*]]
        Tsurf_h     = [reform(line_dum[23,*]),Tsurf_h[index_h:*]]
        Haws_h      = [reform(line_dum[24,*]),Haws_h[index_h:*]]
        Hstk_h      = [reform(line_dum[25,*]),Hstk_h[index_h:*]]
        Hpt_h       = [reform(line_dum[26,*]),Hpt_h[index_h:*]]
        Hpt_cor_h   = [reform(line_dum[27,*]),Hpt_cor_h[index_h:*]]
        Tice1_h     = [reform(line_dum[28,*]),Tice1_h[index_h:*]]
        Tice2_h     = [reform(line_dum[29,*]),Tice2_h[index_h:*]]
        Tice3_h     = [reform(line_dum[30,*]),Tice3_h[index_h:*]]
        Tice4_h     = [reform(line_dum[31,*]),Tice4_h[index_h:*]]
        Tice5_h     = [reform(line_dum[32,*]),Tice5_h[index_h:*]]
        Tice6_h     = [reform(line_dum[33,*]),Tice6_h[index_h:*]]
        Tice7_h     = [reform(line_dum[34,*]),Tice7_h[index_h:*]]
        Tice8_h     = [reform(line_dum[35,*]),Tice8_h[index_h:*]]
        tiltX_h     = [reform(line_dum[36,*]),tiltX_h[index_h:*]]
        tiltY_h     = [reform(line_dum[37,*]),tiltY_h[index_h:*]]
        GPStime_h   = [reform(line_dum[38,*]),GPStime_h[index_h:*]]
        GPSlat_h    = [reform(line_dum[39,*]),GPSlat_h[index_h:*]]
        GPSlon_h    = [reform(line_dum[40,*]),GPSlon_h[index_h:*]]
        GPSelev_h   = [reform(line_dum[41,*]),GPSelev_h[index_h:*]]
        GPShdop_h   = [reform(line_dum[42,*]),GPShdop_h[index_h:*]]
        Tlog_h      = [reform(line_dum[43,*]),Tlog_h[index_h:*]]
        Ifan_h      = [reform(line_dum[44,*]),Ifan_h[index_h:*]]
        Vbat_h      = [reform(line_dum[45,*]),Vbat_h[index_h:*]]
        month_cent_h= (year_h-2000)*12+month_h
        day_cent_d  = fix(min(day_cent_h))+ indgen(fix(max(day_cent_h))-fix(min(day_cent_h))+1)
        WSx_h= WD_h
        WSy_h= WS_h
        WSx_h[*]= -999
        WSy_h[*]= -999
        if total(where(WS_h ne -999)) then WSx_h[where(WS_h ne -999)] = WS_h[where(WS_h ne -999)]*sin(WD_h[where(WS_h ne -999)]*!pi/180.)
        if total(where(WS_h ne -999)) then WSy_h[where(WS_h ne -999)] = WS_h[where(WS_h ne -999)]*cos(WD_h[where(WS_h ne -999)]*!pi/180.)
        month_cent_m = min(month_cent_h) + indgen(max(month_cent_h)-min(month_cent_h)+1)
        if total(where(WSx_h ne -999)) ne -1 then WD_h[where(WSx_h ne -999)] = 180./!pi*atan(WSx_h[where(WSx_h ne -999)]/WSy_h[where(WSx_h ne -999)])
        if total(where(WSy_h lt 0 and WSy_h ne -999)) ne -1 then WD_h[where(WSy_h lt 0 and WSy_h ne -999)] = WD_h[where(WSy_h lt 0 and WSy_h ne -999)]-180
        if total(where(WD_h lt 0 and WD_h ne -999)) ne -1 then WD_h[where(WD_h lt 0 and WD_h ne -999)] = WD_h[where(WD_h lt 0 and WD_h ne -999)]+360
        if total(where(WSy_h eq 0 and WSx_h gt 0)) ne -1 then WD_h[where(WSy_h eq 0 and WSx_h gt 0)] = 90
        if total(where(WSy_h eq 0 and WSx_h lt 0)) ne -1 then WD_h[where(WSy_h eq 0 and WSx_h lt 0)] = 270

        Hpt_cor_h_smooth = smooth(Hpt_cor_h,5)
;print,Hpt_cor_h_smooth
     endif
;stop
; Daily
     p_d = fltarr(n_elements(day_cent_d))
     p_d[*] = -999
     T_d = p_d
     Thc_d = p_d
;RH_d = p_d
     RH_cor_d = p_d
     q_d = p_d
     WSx_d = p_d
     WSy_d = p_d
     WS_d = p_d
     WD_d = p_d
     SHF_d = p_d
     LHF_d = p_d
     SRin_d = p_d
     SRin_cor_d = p_d
     SRout_d = p_d
     SRout_cor_d = p_d
     albedo_d = p_d
     LRin_d = p_d
     LRout_d = p_d
     CloudCov_d = p_d
     Tsurf_d = p_d
     Haws_d = p_d
     Hstk_d = p_d
     Hpt_d = p_d
     Hpt_cor_d = p_d
     ablation_pt_d = p_d
     Tice1_d = p_d
     Tice2_d = p_d
     Tice3_d = p_d
     Tice4_d = p_d
     Tice5_d = p_d
     Tice6_d = p_d
     Tice7_d = p_d
     Tice8_d = p_d
     tiltX_d = p_d
     tiltY_d = p_d
     GPSlat_d = double(p_d)
     GPSlon_d = double(p_d)
     GPSelev_d = p_d
     GPShdop_d = p_d
     Tlog_d = p_d
     Ifan_d = p_d
     Vbat_d = p_d
     for i=0l,n_elements(day_cent_d)-1 do begin
        sel = where(day_cent eq day_cent_d[i] and datatype eq 'd') ;  first fill the arrays with available transmitted data
        if total(sel) ne -1 then begin
           p_d[i] = p[sel[0]]
           T_d[i] = T[sel[0]]
           Thc_d[i] = Thc[sel[0]]
           RH_cor_d[i] = RH_cor[sel[0]]
;    RH_d[i] = RH[sel[0]]
           WS_d[i] = WS[sel[0]]
           WSx_d[i] = WSx[sel[0]]
           WSy_d[i] = WSy[sel[0]]
           WD_d[i] = WD[sel[0]]
           SRin_d[i] = SRin[sel[0]]
           SRout_d[i] = SRout[sel[0]]
           LRin_d[i] = LRin[sel[0]]
           LRout_d[i] = LRout[sel[0]]
           Haws_d[i] = Haws[sel[0]]
           Hstk_d[i] = Hstk[sel[0]]
           Hpt_d[i] = Hpt[sel[0]]
           Hpt_cor_d[i] = Hpt_cor[sel[0]]
           ablation_pt_d[i] = -999 ; Can't calculate daily ablation from daily mean values
           Tice1_d[i] = Tice1[sel[0]]
           Tice2_d[i] = Tice2[sel[0]]
           Tice3_d[i] = Tice3[sel[0]]
           Tice4_d[i] = Tice4[sel[0]]
           Tice5_d[i] = Tice5[sel[0]]
           Tice6_d[i] = Tice6[sel[0]]
           Tice7_d[i] = Tice7[sel[0]]
           Tice8_d[i] = Tice8[sel[0]]
           tiltX_d[i] = tiltX[sel[0]]
           tiltY_d[i] = tiltY[sel[0]]
           if GPShdop[sel[0]] le MinHDOP then GPSlat_d[i]  = GPSlat[sel[0]]
           if GPShdop[sel[0]] le MinHDOP then GPSlon_d[i]  = GPSlon[sel[0]]
           if GPShdop[sel[0]] le MinHDOP then GPSelev_d[i] = GPSelev[sel[0]]
           if GPShdop[sel[0]] le MinHDOP then GPShdop_d[i] = GPShdop[sel[0]]
           Tlog_d[i] = Tlog[sel[0]]
           Ifan_d[i] = Ifan[sel[0]]
           Vbat_d[i] = Vbat[sel[0]]
        endif
        
        sel = where(day_cent_h eq day_cent_d[i]) ; this should overwrite the transmitted daily values in case of overlap
        if total(sel) ne -1 then begin
           if n_elements(where(p_h[sel] ne -999)) ge MinCov*24. then p_d[i] = mean(p_h[sel[where(p_h[sel] ne -999)]])
           if n_elements(where(T_h[sel] ne -999)) ge MinCov*24. then T_d[i] = mean(T_h[sel[where(T_h[sel] ne -999)]])
           if n_elements(where(Thc_h[sel] ne -999)) ge MinCov*24. then Thc_d[i] = mean(Thc_h[sel[where(Thc_h[sel] ne -999)]])
;    if n_elements(where(RH_h[sel] ne -999)) ge MinCov*24. then RH_d[i] = mean(RH_h[sel[where(RH_h[sel] ne -999)]])
           if n_elements(where(RH_cor_h[sel] ne -999)) ge MinCov*24. then RH_cor_d[i] = mean(RH_cor_h[sel[where(RH_cor_h[sel] ne -999)]])
           if n_elements(where(q_h[sel] ne -999)) ge MinCov*24. then q_d[i] = mean(q_h[sel[where(q_h[sel] ne -999)]])
           if n_elements(where(WS_h[sel] ne -999)) ge MinCov*24. then WS_d[i] = mean(WS_h[sel[where(WS_h[sel] ne -999)]])
           if n_elements(where(WSx_h[sel] ne -999)) ge MinCov*24. then WSx_d[i] = mean(WSx_h[sel[where(WSx_h[sel] ne -999)]])
           if n_elements(where(WSy_h[sel] ne -999)) ge MinCov*24. then WSy_d[i] = mean(WSy_h[sel[where(WSy_h[sel] ne -999)]])
           if n_elements(where(SHF_h[sel] ne -999)) ge MinCov*24. then SHF_d[i] = mean(SHF_h[sel[where(SHF_h[sel] ne -999)]])
           if n_elements(where(LHF_h[sel] ne -999)) ge MinCov*24. then LHF_d[i] = mean(LHF_h[sel[where(LHF_h[sel] ne -999)]])
           if n_elements(where(SRin_h[sel] ne -999)) ge MinCov*24. then SRin_d[i] = mean(SRin_h[sel[where(SRin_h[sel] ne -999)]])
           if n_elements(where(SRin_cor_h[sel] ne -999)) ge MinCov*24. then SRin_cor_d[i] = mean(SRin_cor_h[sel[where(SRin_cor_h[sel] ne -999)]])
           if n_elements(where(SRout_h[sel] ne -999)) ge MinCov*24. then SRout_d[i] = mean(SRout_h[sel[where(SRout_h[sel] ne -999)]])
           if n_elements(where(SRout_cor_h[sel] ne -999)) ge MinCov*24. then SRout_cor_d[i] = mean(SRout_cor_h[sel[where(SRout_cor_h[sel] ne -999)]])
           if total(where(albedo_h[sel] ne -999)) ne -1 then albedo_d[i] = mean(albedo_h[sel[where(albedo_h[sel] ne -999)]])
           if n_elements(where(LRin_h[sel] ne -999)) ge MinCov*24. then LRin_d[i] = mean(LRin_h[sel[where(LRin_h[sel] ne -999)]])
           if n_elements(where(LRout_h[sel] ne -999)) ge MinCov*24. then LRout_d[i] = mean(LRout_h[sel[where(LRout_h[sel] ne -999)]])
           if n_elements(where(CloudCov_h[sel] ne -999)) ge MinCov*24. then CloudCov_d[i] = mean(CloudCov_h[sel[where(CloudCov_h[sel] ne -999)]])
           if n_elements(where(Tsurf_h[sel] ne -999)) ge MinCov*24. then Tsurf_d[i] = mean(Tsurf_h[sel[where(Tsurf_h[sel] ne -999)]])
           if total(where(Haws_h[sel] ne -999)) ne -1 then Haws_d[i] = mean(Haws_h[sel[where(Haws_h[sel] ne -999)]])
           if total(where(Hstk_h[sel] ne -999)) ne -1 then Hstk_d[i] = mean(Hstk_h[sel[where(Hstk_h[sel] ne -999)]])
           if total(where(Hpt_h[sel] ne -999)) ne -1 then Hpt_d[i] = mean(Hpt_h[sel[where(Hpt_h[sel] ne -999)]])
           if total(where(Hpt_cor_h[sel] ne -999)) ne -1 then Hpt_cor_d[i] = mean(Hpt_cor_h[sel[where(Hpt_cor_h[sel] ne -999)]])
           if total(where(Tice1_h[sel] ne -999)) ne -1 then Tice1_d[i] = mean(Tice1_h[sel[where(Tice1_h[sel] ne -999)]])
           if total(where(Tice2_h[sel] ne -999)) ne -1 then Tice2_d[i] = mean(Tice2_h[sel[where(Tice2_h[sel] ne -999)]])
           if total(where(Tice3_h[sel] ne -999)) ne -1 then Tice3_d[i] = mean(Tice3_h[sel[where(Tice3_h[sel] ne -999)]])
           if total(where(Tice4_h[sel] ne -999)) ne -1 then Tice4_d[i] = mean(Tice4_h[sel[where(Tice4_h[sel] ne -999)]])
           if total(where(Tice5_h[sel] ne -999)) ne -1 then Tice5_d[i] = mean(Tice5_h[sel[where(Tice5_h[sel] ne -999)]])
           if total(where(Tice6_h[sel] ne -999)) ne -1 then Tice6_d[i] = mean(Tice6_h[sel[where(Tice6_h[sel] ne -999)]])
           if total(where(Tice7_h[sel] ne -999)) ne -1 then Tice7_d[i] = mean(Tice7_h[sel[where(Tice7_h[sel] ne -999)]])
           if total(where(Tice8_h[sel] ne -999)) ne -1 then Tice8_d[i] = mean(Tice8_h[sel[where(Tice8_h[sel] ne -999)]])
           if total(where(tiltX_h[sel] ne -999)) ne -1 then tiltX_d[i] = mean(tiltX_h[sel[where(tiltX_h[sel] ne -999)]])
           if total(where(tiltY_h[sel] ne -999)) ne -1 then tiltY_d[i] = mean(tiltY_h[sel[where(tiltY_h[sel] ne -999)]])
           if total(where(GPSlat_h[sel] ne -999 and GPShdop_h[sel] le MinHDOP)) ne -1 then GPSlat_d[i] = mean(GPSlat_h[sel[where(GPSlat_h[sel] ne -999 and GPShdop_h[sel] le MinHDOP)]])
           if total(where(GPSlon_h[sel] ne -999 and GPShdop_h[sel] le MinHDOP)) ne -1 then GPSlon_d[i] = mean(GPSlon_h[sel[where(GPSlon_h[sel] ne -999 and GPShdop_h[sel] le MinHDOP)]])
           if total(where(GPSelev_h[sel] ne -999 and GPShdop_h[sel] le MinHDOP)) ne -1 then GPSelev_d[i] = mean(GPSelev_h[sel[where(GPSelev_h[sel] ne -999 and GPShdop_h[sel] le MinHDOP)]])
           if total(where(GPShdop_h[sel] ne -999 and GPShdop_h[sel] le MinHDOP)) ne -1 then GPShdop_d[i] = mean(GPShdop_h[sel[where(GPShdop_h[sel] ne -999 and GPShdop_h[sel] le MinHDOP)]])
           if total(where(Tlog_h[sel] ne -999)) ne -1 then Tlog_d[i] = mean(Tlog_h[sel[where(Tlog_h[sel] ne -999)]])
           if total(where(Ifan_h[sel] ne -999)) ne -1 then Ifan_d[i] = mean(Ifan_h[sel[where(Ifan_h[sel] ne -999)]])
           if total(where(Vbat_h[sel] ne -999)) ne -1 then Vbat_d[i] = mean(Vbat_h[sel[where(Vbat_h[sel] ne -999)]])
        endif
        sel2 = where(day_cent_h eq day_cent_d[i] and hour_h eq 23)
        if i ne 0 and total(sel2) ne -1 then begin
;    if Hpt_cor_h[sel2] ne -999 and Hpt_cor_h_prevday ne -999 and Hpt_cor_h_prevday-Hpt_cor_h[sel2] ge 0 and Hpt_cor_h_prevday-Hpt_cor_h[sel2] lt 0.2 then ablation_pt_d[i] = 1000*(Hpt_cor_h_prevday - Hpt_cor_h[sel2])
           if Hpt_cor_h_smooth[sel2] ne -999 and Hpt_cor_h_prevday ne -999 and Hpt_cor_h_prevday-Hpt_cor_h_smooth[sel2] ge 0 and Hpt_cor_h_prevday-Hpt_cor_h_smooth[sel2] lt 0.2 then ablation_pt_d[i] = 1000*(Hpt_cor_h_prevday - Hpt_cor_h_smooth[sel2])
;    if Hpt_cor_h_smooth[sel2] ne -999 and Hpt_cor_h_prevday ne -999 and Hpt_cor_h_prevday-Hpt_cor_h_smooth[sel2] ge 0 and Hpt_cor_h_prevday-Hpt_cor_h_smooth[sel2] lt 0.2 then ablation_pt_d[i] = 1000*(Hpt_cor_h_prevday - Hpt_cor_h_smooth[sel2])
;    Hpt_cor_h_prevday = Hpt_cor_h[sel2];Hpt_cor_h_smooth[sel2]
           Hpt_cor_h_prevday = Hpt_cor_h_smooth[sel2]
        endif
        if i eq 0 then Hpt_cor_h_prevday = -999
     endfor

     caldat,day_cent_d+2451544,month_d,day_d,year_d
     day_year_d = JulDay(month_d,day_d,year_d)-JulDay(12,31,year_d-1)
     if total(where(WSx_d ne -999)) ne -1 then WD_d[where(WSx_d ne -999)] = 180./!pi*atan(WSx_d[where(WSx_d ne -999)]/WSy_d[where(WSx_d ne -999)])
     if total(where(WSy_d lt 0 and WSy_d ne -999)) ne -1 then WD_d[where(WSy_d lt 0 and WSy_d ne -999)] = WD_d[where(WSy_d lt 0 and WSy_d ne -999)]-180
     if total(where(WD_d lt 0 and WD_d ne -999)) ne -1 then WD_d[where(WD_d lt 0 and WD_d ne -999)] = WD_d[where(WD_d lt 0 and WD_d ne -999)]+360
     if total(where(WSy_d eq 0 and WSx_d gt 0)) ne -1 then WD_d[where(WSy_d eq 0 and WSx_d gt 0)] = 90
     if total(where(WSy_d eq 0 and WSx_d lt 0)) ne -1 then WD_d[where(WSy_d eq 0 and WSx_d lt 0)] = 270

     if updaterun eq 'yes' then begin
        b_t=0
        columns1 = columns_day  ;45 ;day
        filename_old = dir+datadir+AWS[i_AWS]+'_day'+version_no+'.txt'
        nlines_old=FILE_LINES(filename_old)
        line_old = fltarr(columns1,nlines_old-1)
        OPENR, lun, filename_old, /GET_LUN
        header = STRARR(1)
        READF, lun, header      ; Read one line at a time, saving the result into array
        READF, lun, line_old    ;_full
        CLOSE, lun
        free_lun,lun
        cen_old=line_old(4,0)
        index_d=0
        cen_new=(min(day_cent[where(day_cent gt 0)])) ;*24;line_upd(5,0)
        FOR i=0,nlines_old-2 DO BEGIN
           IF (fix(line_old(4,i)) eq fix(cen_new) ) THEN b_t=i ; and fix(line_old(3,i)) eq fix(cen_new)
        ENDFOR
        line_dum=fltarr(columns1,b_t)
        line_dum=line_old[*,0:b_t]

        p_d[0:b_t]            =[reform(line_dum[5,*])]
        T_d[0:b_t]            =[reform(line_dum[6,*])]
        Thc_d[0:b_t]          = [reform(line_dum[7,*])]
                                ;RH_d[0:b_t]           = [reform(line_dum[8,*])]
        RH_cor_d[0:b_t]       = [reform(line_dum[8,*])]
        q_d[0:b_t]            = [reform(line_dum[9,*])]
        WS_d[0:b_t]           = [reform(line_dum[10,*])]
        WD_d[0:b_t]           = [reform(line_dum[11,*])]
        SHF_d[0:b_t]          = [reform(line_dum[12,*])]
        LHF_d[0:b_t]          = [reform(line_dum[13,*])]
        SRin_d[0:b_t]         = [reform(line_dum[14,*])]
        SRin_cor_d[0:b_t]     = [reform(line_dum[15,*])]
        SRout_d[0:b_t]        = [reform(line_dum[16,*])]
        SRout_cor_d[0:b_t]    = [reform(line_dum[17,*])]
        albedo_d[0:b_t]       = [reform(line_dum[18,*])]
        LRin_d[0:b_t]         = [reform(line_dum[19,*])]
        LRout_d[0:b_t]        = [reform(line_dum[20,*])]
        CloudCov_d[0:b_t]     = [reform(line_dum[21,*])]
        Tsurf_d[0:b_t]        = [reform(line_dum[22,*])]
        Haws_d[0:b_t]         = [reform(line_dum[23,*])]
        Hstk_d[0:b_t]         = [reform(line_dum[24,*])]
        Hpt_d[0:b_t]          = [reform(line_dum[25,*])]
        Hpt_cor_d[0:b_t]      = [reform(line_dum[26,*])]
        ablation_pt_d[0:b_t]  = [reform(line_dum[27,*])]
        Tice1_d[0:b_t]        = [reform(line_dum[28,*])]
        Tice2_d[0:b_t]        = [reform(line_dum[29,*])]
        Tice3_d[0:b_t]        = [reform(line_dum[30,*])]
        Tice4_d[0:b_t]        = [reform(line_dum[31,*])]
        Tice5_d[0:b_t]        = [reform(line_dum[32,*])]
        Tice6_d[0:b_t]        = [reform(line_dum[33,*])]
        Tice7_d[0:b_t]        = [reform(line_dum[34,*])]
        Tice8_d[0:b_t]        = [reform(line_dum[35,*])]
        tiltX_d[0:b_t]        = [reform(line_dum[36,*])]
        tiltY_d[0:b_t]        = [reform(line_dum[37,*])]
        GPSlat_d[0:b_t]       = [reform(line_dum[38,*])]
        GPSlon_d[0:b_t]       = [reform(line_dum[39,*])]
        GPSelev_d[0:b_t]      = [reform(line_dum[40,*])]
        GPShdop_d[0:b_t]      = [reform(line_dum[41,*])]
        Tlog_d[0:b_t]         = [reform(line_dum[42,*])]
        Ifan_d[0:b_t]         = [reform(line_dum[43,*])]
        Vbat_d[0:b_t]         = [reform(line_dum[44,*])]
        day_cent_d  = fix(min(day_cent_d))+ indgen(fix(max(day_cent_d))-fix(min(day_cent_d))+1)
        WSx_d= WD_d
        WSy_d= WS_d
        WSx_d[*]= -999
        WSy_d[*]= -999
        if total(where(WS_d ne -999)) then WSx_d[where(WS_d ne -999)] = WS_d[where(WS_d ne -999)]*sin(WD_d[where(WS_d ne -999)]*!pi/180.)
        if total(where(WS_d ne -999)) then WSy_d[where(WS_d ne -999)] = WS_d[where(WS_d ne -999)]*cos(WD_d[where(WS_d ne -999)]*!pi/180.)
        if total(where(WSx_d ne -999)) ne -1 then WD_d[where(WSx_d ne -999)] = 180./!pi*atan(WSx_d[where(WSx_d ne -999)]/WSy_d[where(WSx_d ne -999)])
        if total(where(WSy_d lt 0 and WSy_d ne -999)) ne -1 then WD_d[where(WSy_d lt 0 and WSy_d ne -999)] = WD_d[where(WSy_d lt 0 and WSy_d ne -999)]-180
        if total(where(WD_d lt 0 and WD_d ne -999)) ne -1 then WD_d[where(WD_d lt 0 and WD_d ne -999)] = WD_d[where(WD_d lt 0 and WD_d ne -999)]+360
        if total(where(WSy_d eq 0 and WSx_d gt 0)) ne -1 then WD_d[where(WSy_d eq 0 and WSx_d gt 0)] = 90
        if total(where(WSy_d eq 0 and WSx_d lt 0)) ne -1 then WD_d[where(WSy_d eq 0 and WSx_d lt 0)] = 270

     endif

; Monthly
     p_m = fltarr(n_elements(month_cent_m))
     p_m[*] = -999
     T_m = p_m
     Thc_m = p_m
     RH_cor_m = p_m
     q_m = p_m
     WSx_m = p_m
     WSy_m = p_m
     WS_m = p_m
     WD_m = p_m
     SHF_m = p_m
     LHF_m = p_m
     SRin_m = p_m
     SRin_cor_m = p_m
     SRout_m = p_m
     SRout_cor_m = p_m
     LRin_m = p_m
     LRout_m = p_m
     albedo_m = p_m
     Haws_m = p_m
     Hstk_m = p_m
     Hpt_m = p_m
     Hpt_cor_m = p_m
     GPSlat_m = double(p_m)
     GPSlon_m = double(p_m)
     GPSelev_m = p_m
     fanOK_m = p_m
     for i=0l,n_elements(month_cent_m)-1 do begin
        sel = where(month_d+(year_d-2000)*12 eq month_cent_m[i])
        if total(sel) ne -1 then begin
           if n_elements(where(p_d[sel] ne -999)) ge MinDays then p_m[i] = mean(p_d[sel[where(p_d[sel] ne -999)]])
           if n_elements(where(T_d[sel] ne -999)) ge MinDays then T_m[i] = mean(T_d[sel[where(T_d[sel] ne -999)]])
           if n_elements(where(Thc_d[sel] ne -999)) ge MinDays then Thc_m[i] = mean(Thc_d[sel[where(Thc_d[sel] ne -999)]])
           if n_elements(where(RH_cor_d[sel] ne -999)) ge MinDays then RH_cor_m[i] = mean(RH_cor_d[sel[where(RH_cor_d[sel] ne -999)]])
           if n_elements(where(q_d[sel] ne -999)) ge MinDays then q_m[i] = mean(q_d[sel[where(q_d[sel] ne -999)]])
           if n_elements(where(WS_d[sel] ne -999)) ge MinDays then WS_m[i] = mean(WS_d[sel[where(WS_d[sel] ne -999)]])
           if n_elements(where(WSx_d[sel] ne -999)) ge MinDays then WSx_m[i] = mean(WSx_d[sel[where(WSx_d[sel] ne -999)]])
           if n_elements(where(WSy_d[sel] ne -999)) ge MinDays then WSy_m[i] = mean(WSy_d[sel[where(WSy_d[sel] ne -999)]])
           if n_elements(where(SHF_d[sel] ne -999)) ge MinDays then SHF_m[i] = mean(SHF_d[sel[where(SHF_d[sel] ne -999)]])
           if n_elements(where(LHF_d[sel] ne -999)) ge MinDays then LHF_m[i] = mean(LHF_d[sel[where(LHF_d[sel] ne -999)]])
           if n_elements(where(SRin_d[sel] ne -999)) ge MinDays then SRin_m[i] = mean(SRin_d[sel[where(SRin_d[sel] ne -999)]])
           if n_elements(where(SRin_cor_d[sel] ne -999)) ge MinDays then SRin_cor_m[i] = mean(SRin_cor_d[sel[where(SRin_cor_d[sel] ne -999)]])
           if n_elements(where(SRout_d[sel] ne -999)) ge MinDays then SRout_m[i] = mean(SRout_d[sel[where(SRout_d[sel] ne -999)]])
           if n_elements(where(SRout_cor_d[sel] ne -999)) ge MinDays then SRout_cor_m[i] = mean(SRout_cor_d[sel[where(SRout_cor_d[sel] ne -999)]])
           if n_elements(where(albedo_d[sel] ne -999)) ge MinDays then albedo_m[i] = mean(albedo_d[sel[where(albedo_d[sel] ne -999)]])
           if n_elements(where(LRin_d[sel] ne -999)) ge MinDays then LRin_m[i] = mean(LRin_d[sel[where(LRin_d[sel] ne -999)]])
           if n_elements(where(LRout_d[sel] ne -999)) ge MinDays then LRout_m[i] = mean(LRout_d[sel[where(LRout_d[sel] ne -999)]])
           if n_elements(where(Haws_d[sel] ne -999)) ge MinDays then Haws_m[i] = mean(Haws_d[sel[where(Haws_d[sel] ne -999)]])
;    if n_elements(where(Hstk_d[sel] ne -999)) ge MinDays then Hstk_m[i] = mean(Hstk_d[sel[where(Hstk_d[sel] ne -999)]])
;    if n_elements(where(Hpt_d[sel] ne -999)) ge MinDays then Hpt_m[i] = mean(Hpt_d[sel[where(Hpt_d[sel] ne -999)]])
;    if n_elements(where(Hpt_cor_d[sel] ne -999)) ge MinDays then Hpt_cor_m[i] = mean(Hpt_cor_d[sel[where(Hpt_cor_d[sel] ne -999)]])
           if n_elements(where(GPSlat_d[sel] ne -999)) ge MinDays then GPSlat_m[i] = mean(GPSlat_d[sel[where(GPSlat_d[sel] ne -999)]])
           if n_elements(where(GPSlon_d[sel] ne -999)) ge MinDays then GPSlon_m[i] = mean(GPSlon_d[sel[where(GPSlon_d[sel] ne -999)]])
           if n_elements(where(GPSelev_d[sel] ne -999)) ge MinDays then GPSelev_m[i] = mean(GPSelev_d[sel[where(GPSelev_d[sel] ne -999)]])
           if total(where(Ifan_d[sel] gt 50)) ne -1 then fanOK_m[i] = 100.*n_elements(where(Ifan_d[sel] gt 50))/n_elements(where(Ifan_d[sel] ne -999))
        endif
     endfor
     month_m = round(((month_cent_m-1)/12. - (fix(month_cent_m)-1)/12)*12) + 1
     year_m = (month_cent_m-1)/12 + 2000
     if total(where(WSx_m ne -999)) ne -1 then WD_m[where(WSx_m ne -999)] = 180./!pi*atan(WSx_m[where(WSx_m ne -999)]/WSy_m[where(WSx_m ne -999)])
     if total(where(WSy_m lt 0 and WSy_m ne -999)) ne -1 then WD_m[where(WSy_m lt 0 and WSy_m ne -999)] = WD_m[where(WSy_m lt 0 and WSy_m ne -999)]-180
     if total(where(WD_m lt 0 and WD_m ne -999)) ne -1 then WD_m[where(WD_m lt 0 and WD_m ne -999)] = WD_m[where(WD_m lt 0 and WD_m ne -999)]+360
     if total(where(WSy_m eq 0 and WSx_m gt 0)) ne -1 then WD_m[where(WSy_m eq 0 and WSx_m gt 0)] = 90
     if total(where(WSy_m eq 0 and WSx_m lt 0)) ne -1 then WD_m[where(WSy_m eq 0 and WSx_m lt 0)] = 270
;if total(where(fanOK_m gt 100)) ne -1 then fanOK_m[where(fanOK_m gt 100)] = 100 ; just to make sure that months with more than 28 days don't give fanOK > 100%
     if updaterun eq 'yes' then begin
        c_t=0
        columns1 = columns_mon  ;24 ;day
        filename_old = dir+datadir+AWS[i_AWS]+'_month'+version_no+'.txt'
        nlines_old=FILE_LINES(filename_old)
        line_old = fltarr(columns1,nlines_old-1)
        OPENR, lun, filename_old, /GET_LUN
        header = STRARR(1)
        READF, lun, header      ; Read one line at a time, saving the result into array
        READF, lun, line_old    ;_full
        CLOSE, lun
        free_lun,lun
        cen_old=line_old(2,0)
        index_m=0
        cen_new=(min(month_cent[where(month_cent gt 0)])) ;*24;line_upd(5,0)
        FOR i=0,nlines_old-2 DO BEGIN
           IF (fix(line_old(2,i)) eq fix(cen_new) ) THEN c_t=i ; and fix(line_old(3,i)) eq fix(cen_new)
        ENDFOR
        line_dum=fltarr(columns1,c_t)
        line_dum=line_old[*,0:c_t]

        p_m[0:c_t]         = [reform(line_dum[3,*])]
        T_m[0:c_t]         = [reform(line_dum[4,*])]
        Thc_m[0:c_t]       = [reform(line_dum[5,*])]
        RH_cor_m[0:c_t]    = [reform(line_dum[6,*])]
        q_m[0:c_t]         = [reform(line_dum[7,*])]
        WS_m[0:c_t]        = [reform(line_dum[8,*])]
        WD_m[0:c_t]        = [reform(line_dum[9,*])]
        SHF_m[0:c_t]       = [reform(line_dum[10,*])]
        LHF_m[0:c_t]       = [reform(line_dum[11,*])]
        SRin_m[0:c_t]      = [reform(line_dum[12,*])]
        SRin_cor_m[0:c_t]  = [reform(line_dum[13,*])]
        SRout_m[0:c_t]     = [reform(line_dum[14,*])]
        SRout_cor_m[0:c_t] = [reform(line_dum[15,*])]
        albedo_m[0:c_t]    = [reform(line_dum[16,*])]
        LRin_m[0:c_t]      = [reform(line_dum[17,*])]
        LRout_m[0:c_t]     = [reform(line_dum[18,*])]
        Haws_m[0:c_t]      = [reform(line_dum[19,*])]
        GPSlat_m[0:c_t]    = [reform(line_dum[20,*])]
        GPSlon_m[0:c_t]    = [reform(line_dum[21,*])]
        GPSelev_m[0:c_t]   = [reform(line_dum[22,*])]
        fanOK_m[0:c_t]     = [reform(line_dum[23,*])]

        WSx_m              = WS_m 
        WSy_m              = WS_m
        WSx_m[*]           = -999
        WSy_m[*]           = -999
        if total(where(WSx_m ne -999)) ne -1 then WD_m[where(WSx_m ne -999)] = 180./!pi*atan(WSx_m[where(WSx_m ne -999)]/WSy_m[where(WSx_m ne -999)])
        if total(where(WSy_m lt 0 and WSy_m ne -999)) ne -1 then WD_m[where(WSy_m lt 0 and WSy_m ne -999)] = WD_m[where(WSy_m lt 0 and WSy_m ne -999)]-180
        if total(where(WD_m lt 0 and WD_m ne -999)) ne -1 then WD_m[where(WD_m lt 0 and WD_m ne -999)] = WD_m[where(WD_m lt 0 and WD_m ne -999)]+360
        if total(where(WSy_m eq 0 and WSx_m gt 0)) ne -1 then WD_m[where(WSy_m eq 0 and WSx_m gt 0)] = 90
        if total(where(WSy_m eq 0 and WSx_m lt 0)) ne -1 then WD_m[where(WSy_m eq 0 and WSx_m lt 0)] = 270

     endif
;----------------------------------------------------------------------------------------------------
; Writing data to file ------------------------------------------------------------------------------
     if inst_output eq 'yes' then begin
        format_i = '(6i5,3f8.2,2f7.1,f8.2,5f7.1,f9.3,2f7.1,3f8.2,f9.3,i5,f9.3,i5,2f9.3,10f8.2,i7,2f13.7,f7.1,2f8.2,f7.1,f8.2,f7.1,i5,f7.1,i5,f7.1,f8.2)'
        if updaterun eq 'yes' then begin
           openw,lun,dir+datadir+AWS[i_AWS]+'_inst'+version_no+'_upd.txt',/get_lun
        endif else begin
           openw,lun,dir+datadir+AWS[i_AWS]+'_inst'+version_no+'.txt',/get_lun
        endelse
        printf,lun,' Year MonthOfYear DayOfMonth HourOfDay(UTC) MinuteOfHour DayOfYear'+ $
               ' AirPressure(hPa) AirTemperature(C) AirTemperatureHygroClip(C) RelativeHumidity_wrtWater(%) RelativeHumidity(%)'+ $
               ' WindSpeed(m/s) WindDirection(d) ShortwaveRadiationDown(W/m2) ShortwaveRadiationDown_Cor(W/m2)'+ $
               ' ShortwaveRadiationUp(W/m2) ShortwaveRadiationUp_Cor(W/m2) Albedo LongwaveRadiationDown(W/m2) LongwaveRadiationUp(W/m2)'+ $
               ' TemperatureRadiometer(C) CloudCover SurfaceTemperature(C) HeightSensorBoom(m) HeightSensorBoomQuality'+ $
               ' HeightStakes(m) HeightStakesQuality DepthPressureTransducer(m) DepthPressureTransducer_Cor(m) IceTemperature1(C) IceTemperature2(C)'+ $
               ' IceTemperature3(C) IceTemperature4(C) IceTemperature5(C) IceTemperature6(C) IceTemperature7(C) IceTemperature8(C)'+ $
               ' TiltToEast(d) TiltToNorth(d) TimeGPS(hhmmssUTC) LatitudeGPS(degN) LongitudeGPS(degW) ElevationGPS(m)'+ $
               ' HorDilOfPrecGPS LoggerTemperature(C) FanCurrent(mA) BatteryVoltage(V) TiltAWS(d) TiltDirectionAWS(d)'+ $
               ' ZenithAngleSun(d) DirectionSun(d) AngleBetweenSunAndAWS(d) ShortwaveRadiationDownCorrectionFactor'
        for i=1l,lines_data_total-1 do printf,lun,format=format_i,year[i],month[i],day[i],hour[i],minute[i],day_year[i], $
                                              p[i],T[i],Thc[i],RH[i],RH_cor[i],WS[i],WD[i],SRin[i],SRin_cor[i],SRout[i],SRout_cor[i],albedo[i], $
                                              LRin[i],LRout[i],Trad[i],CloudCov[i],Tsurf[i],Haws[i],Haws_qual[i],Hstk[i],Hstk_qual[i],Hpt[i],Hpt_cor[i], $
                                              Tice1[i],Tice2[i],Tice3[i],Tice4[i],Tice5[i],Tice6[i],Tice7[i],Tice8[i],tiltX[i],tiltY[i], $
                                              GPStime[i],GPSlat[i],GPSlon[i],GPSelev[i],GPShdop[i],Tlog[i],Ifan[i],Vbat[i], $
                                              theta_sensor_deg[i],phi_sensor_deg[i],ZenithAngle_deg[i],DirectionSun_deg[i],AngleDif_deg[i],CorFac_all[i]
        free_lun,lun
        print,'done writing inst', AWS[i_AWS]
;endelse
     endif
     print,'total_lines: ', lines_data_total
     format_h = '(6i5,3f8.2,f7.1,2f8.2,7f8.1,f9.3,2f7.1,2f8.2,4f9.3,10f8.2,i7,2f13.7,f7.1,2f8.2,f7.1,f8.2)'
     if updaterun eq 'yes' then begin
;----------Updating hourly data files-----------------------------------------------------------------------
        openw,lun1,dir+datadir+AWS[i_AWS]+'_hour'+version_no+'_upd.txt',/get_lun
        printf,lun1,' Year MonthOfYear DayOfMonth HourOfDay(UTC) DayOfYear DayOfCentury'+ $
               ' AirPressure(hPa) AirTemperature(C) AirTemperatureHygroClip(C) RelativeHumidity(%) SpecificHumidity(g/kg)'+ $
               ' WindSpeed(m/s) WindDirection(d) SensibleHeatFlux(W/m2) LatentHeatFlux(W/m2) ShortwaveRadiationDown(W/m2) ShortwaveRadiationDown_Cor(W/m2)'+ $
               ' ShortwaveRadiationUp(W/m2) ShortwaveRadiationUp_Cor(W/m2) Albedo_theta<70d LongwaveRadiationDown(W/m2) LongwaveRadiationUp(W/m2)'+ $
               ' CloudCover SurfaceTemperature(C) HeightSensorBoom(m)'+ $
               ' HeightStakes(m) DepthPressureTransducer(m) DepthPressureTransducer_Cor(m) IceTemperature1(C) IceTemperature2(C)'+ $
               ' IceTemperature3(C) IceTemperature4(C) IceTemperature5(C) IceTemperature6(C) IceTemperature7(C) IceTemperature8(C)'+ $
               ' TiltToEast(d) TiltToNorth(d) TimeGPS(hhmmssUTC) LatitudeGPS(degN) LongitudeGPS(degW) ElevationGPS(m)'+ $
               ' HorDilOfPrecGPS LoggerTemperature(C) FanCurrent(mA) BatteryVoltage(V)'
        for i=0l,a_t-index_h+n_elements(hour_cent_h)-1 do printf,lun1,format=format_h,year_h[i],month_h[i],day_h[i],hour_h[i],day_year_h[i],day_cent_h[i], $
           p_h[i],T_h[i],Thc_h[i],RH_cor_h[i],q_h[i],WS_h[i],WD_h[i],SHF_h[i],LHF_h[i],SRin_h[i],SRin_cor_h[i],SRout_h[i],SRout_cor_h[i],albedo_h[i], $
           LRin_h[i],LRout_h[i],CloudCov_h[i],Tsurf_h[i],Haws_h[i],Hstk_h[i],Hpt_h[i],Hpt_cor_h[i], $
           Tice1_h[i],Tice2_h[i],Tice3_h[i],Tice4_h[i],Tice5_h[i],Tice6_h[i],Tice7_h[i],Tice8_h[i],tiltX_h[i],tiltY_h[i], $
           GPStime_h[i],GPSlat_h[i],GPSlon_h[i],GPSelev_h[i],GPShdop_h[i],Tlog_h[i],Ifan_h[i],Vbat_h[i]
        free_lun,lun1
     endif else begin
        openw,lun1,dir+datadir+AWS[i_AWS]+'_hour'+version_no+'.txt',/get_lun
        printf,lun1,' Year MonthOfYear DayOfMonth HourOfDay(UTC) DayOfYear DayOfCentury'+ $
               ' AirPressure(hPa) AirTemperature(C) AirTemperatureHygroClip(C) RelativeHumidity(%) SpecificHumidity(g/kg)'+ $
               ' WindSpeed(m/s) WindDirection(d) SensibleHeatFlux(W/m2) LatentHeatFlux(W/m2) ShortwaveRadiationDown(W/m2) ShortwaveRadiationDown_Cor(W/m2)'+ $
               ' ShortwaveRadiationUp(W/m2) ShortwaveRadiationUp_Cor(W/m2) Albedo_theta<70d LongwaveRadiationDown(W/m2) LongwaveRadiationUp(W/m2)'+ $
               ' CloudCover SurfaceTemperature(C) HeightSensorBoom(m)'+ $
               ' HeightStakes(m) DepthPressureTransducer(m) DepthPressureTransducer_Cor(m) IceTemperature1(C) IceTemperature2(C)'+ $
               ' IceTemperature3(C) IceTemperature4(C) IceTemperature5(C) IceTemperature6(C) IceTemperature7(C) IceTemperature8(C)'+ $
               ' TiltToEast(d) TiltToNorth(d) TimeGPS(hhmmssUTC) LatitudeGPS(degN) LongitudeGPS(degW) ElevationGPS(m)'+ $
               ' HorDilOfPrecGPS LoggerTemperature(C) FanCurrent(mA) BatteryVoltage(V)'
        for i=0l,n_elements(hour_cent_h)-1 do printf,lun1,format=format_h,year_h[i],month_h[i],day_h[i],hour_h[i],day_year_h[i],day_cent_h[i], $
           p_h[i],T_h[i],Thc_h[i],RH_cor_h[i],q_h[i],WS_h[i],WD_h[i],SHF_h[i],LHF_h[i],SRin_h[i],SRin_cor_h[i],SRout_h[i],SRout_cor_h[i],albedo_h[i], $
           LRin_h[i],LRout_h[i],CloudCov_h[i],Tsurf_h[i],Haws_h[i],Hstk_h[i],Hpt_h[i],Hpt_cor_h[i], $
           Tice1_h[i],Tice2_h[i],Tice3_h[i],Tice4_h[i],Tice5_h[i],Tice6_h[i],Tice7_h[i],Tice8_h[i],tiltX_h[i],tiltY_h[i], $
           GPStime_h[i],GPSlat_h[i],GPSlon_h[i],GPSelev_h[i],GPShdop_h[i],Tlog_h[i],Ifan_h[i],Vbat_h[i]
        free_lun,lun1
     endelse

     format_d = '(5i5,3f8.2,f7.1,2f8.2,7f8.1,f9.3,2f7.1,2f8.2,4f9.3,i5,10f8.2,2f13.7,f7.1,2f8.2,f7.1,f8.2)'
     if updaterun eq 'yes' then begin
;----------Updating daily data files-----------------------------------------------------------------------
        openw,lun2,dir+datadir+AWS[i_AWS]+'_day'+version_no+'_upd.txt',/get_lun
        printf,lun2,' Year MonthOfYear DayOfMonth DayOfYear DayOfCentury'+ $
               ' AirPressure(hPa) AirTemperature(C) AirTemperatureHygroClip(C) RelativeHumidity(%) SpecificHumidity(g/kg)'+ $
               ' WindSpeed(m/s) WindDirection(d) SensibleHeatFlux(W/m2) LatentHeatFlux(W/m2) ShortwaveRadiationDown(W/m2) ShortwaveRadiationDown_Cor(W/m2)'+ $
               ' ShortwaveRadiationUp(W/m2) ShortwaveRadiationUp_Cor(W/m2) Albedo_theta<70d LongwaveRadiationDown(W/m2) LongwaveRadiationUp(W/m2)'+ $
               ' CloudCover SurfaceTemperature(C) HeightSensorBoom(m)'+ $
               ' HeightStakes(m) DepthPressureTransducer(m) DepthPressureTransducer_Cor(m) AblationPressureTransducer(mm) IceTemperature1(C) IceTemperature2(C)'+ $
               ' IceTemperature3(C) IceTemperature4(C) IceTemperature5(C) IceTemperature6(C) IceTemperature7(C) IceTemperature8(C)'+ $
               ' TiltToEast(d) TiltToNorth(d) LatitudeGPS_HDOP<1(degN) LongitudeGPS_HDOP<1(degW) ElevationGPS_HDOP<1(m)'+ $
               ' HorDilOfPrecGPS_HDOP<1 LoggerTemperature(C) FanCurrent(mA) BatteryVoltage(V)'
        for i=0l,n_elements(day_cent_d)-1 do printf,lun2,format=format_d,year_d[i],month_d[i],day_d[i],day_year_d[i],day_cent_d[i], $
           p_d[i],T_d[i],Thc_d[i],RH_cor_d[i],q_d[i],WS_d[i],WD_d[i],SHF_d[i],LHF_d[i],SRin_d[i],SRin_cor_d[i],SRout_d[i],SRout_cor_d[i],albedo_d[i], $
           LRin_d[i],LRout_d[i],CloudCov_d[i],Tsurf_d[i],Haws_d[i],Hstk_d[i],Hpt_d[i],Hpt_cor_d[i],ablation_pt_d[i], $
           Tice1_d[i],Tice2_d[i],Tice3_d[i],Tice4_d[i],Tice5_d[i],Tice6_d[i],Tice7_d[i],Tice8_d[i],tiltX_d[i],tiltY_d[i], $
           GPSlat_d[i],GPSlon_d[i],GPSelev_d[i],GPShdop_d[i],Tlog_d[i],Ifan_d[i],Vbat_d[i]
        free_lun,lun2
     endif else begin
        openw,lun2,dir+datadir+AWS[i_AWS]+'_day'+version_no+'.txt',/get_lun
        printf,lun2,' Year MonthOfYear DayOfMonth DayOfYear DayOfCentury'+ $
               ' AirPressure(hPa) AirTemperature(C) AirTemperatureHygroClip(C) RelativeHumidity(%) SpecificHumidity(g/kg)'+ $
               ' WindSpeed(m/s) WindDirection(d) SensibleHeatFlux(W/m2) LatentHeatFlux(W/m2) ShortwaveRadiationDown(W/m2) ShortwaveRadiationDown_Cor(W/m2)'+ $
               ' ShortwaveRadiationUp(W/m2) ShortwaveRadiationUp_Cor(W/m2) Albedo_theta<70d LongwaveRadiationDown(W/m2) LongwaveRadiationUp(W/m2)'+ $
               ' CloudCover SurfaceTemperature(C) HeightSensorBoom(m)'+ $
               ' HeightStakes(m) DepthPressureTransducer(m) DepthPressureTransducer_Cor(m) AblationPressureTransducer(mm) IceTemperature1(C) IceTemperature2(C)'+ $
               ' IceTemperature3(C) IceTemperature4(C) IceTemperature5(C) IceTemperature6(C) IceTemperature7(C) IceTemperature8(C)'+ $
               ' TiltToEast(d) TiltToNorth(d) LatitudeGPS_HDOP<1(degN) LongitudeGPS_HDOP<1(degW) ElevationGPS_HDOP<1(m)'+ $
               ' HorDilOfPrecGPS_HDOP<1 LoggerTemperature(C) FanCurrent(mA) BatteryVoltage(V)'
        for i=0l,n_elements(day_cent_d)-1 do printf,lun2,format=format_d,year_d[i],month_d[i],day_d[i],day_year_d[i],day_cent_d[i], $
           p_d[i],T_d[i],Thc_d[i],RH_cor_d[i],q_d[i],WS_d[i],WD_d[i],SHF_d[i],LHF_d[i],SRin_d[i],SRin_cor_d[i],SRout_d[i],SRout_cor_d[i],albedo_d[i], $
           LRin_d[i],LRout_d[i],CloudCov_d[i],Tsurf_d[i],Haws_d[i],Hstk_d[i],Hpt_d[i],Hpt_cor_d[i],ablation_pt_d[i], $
           Tice1_d[i],Tice2_d[i],Tice3_d[i],Tice4_d[i],Tice5_d[i],Tice6_d[i],Tice7_d[i],Tice8_d[i],tiltX_d[i],tiltY_d[i], $
           GPSlat_d[i],GPSlon_d[i],GPSelev_d[i],GPShdop_d[i],Tlog_d[i],Ifan_d[i],Vbat_d[i]
        free_lun,lun2
     endelse
     format_m = '(3i5,3f8.2,f7.1,2f8.2,7f8.1,f9.3,2f7.1,f9.3,2f13.7,f7.1,i5)'
     if updaterun eq 'yes' then begin
;----------Updating monthly data files-----------------------------------------------------------------------
        openw,lun3,dir+datadir+AWS[i_AWS]+'_month'+version_no+'_upd.txt',/get_lun
        printf,lun3,' Year MonthOfYear MonthOfCentury'+ $
               ' AirPressure(hPa) AirTemperature(C) AirTemperatureHygroClip(C) RelativeHumidity(%) SpecificHumidity(g/kg)'+ $
               ' WindSpeed(m/s) WindDirection(d) SensibleHeatFlux(W/m2) LatentHeatFlux(W/m2) ShortwaveRadiationDown(W/m2) ShortwaveRadiationDown_Cor(W/m2)' + $
               ' ShortwaveRadiationUp(W/m2) ShortwaveRadiationUp_Cor(W/m2) Albedo_theta<70d LongwaveRadiationDown(W/m2) LongwaveRadiationUp(W/m2) HeightSensorBoom(m)' + $
               ' LatitudeGPS_HDOP<1(degN) LongitudeGPS_HDOP<1(degW) ElevationGPS_HDOP<1(m) FanOK(%)'
        for i=0,n_elements(month_cent_m)-1 do printf,lun3,format=format_m,year_m[i],month_m[i],month_cent_m[i], $
           p_m[i],T_m[i],Thc_m[i],RH_cor_m[i],q_m[i],WS_m[i],WD_m[i],SHF_m[i],LHF_m[i],SRin_m[i],SRin_cor_m[i],SRout_m[i],SRout_cor_m[i],albedo_m[i], $
           LRin_m[i],LRout_m[i],Haws_m[i],GPSlat_m[i],GPSlon_m[i],GPSelev_m[i],fanOK_m[i]
        free_lun,lun3
     endif else begin
        openw,lun3,dir+datadir+AWS[i_AWS]+'_month'+version_no+'.txt',/get_lun
        printf,lun3,' Year MonthOfYear MonthOfCentury'+ $
               ' AirPressure(hPa) AirTemperature(C) AirTemperatureHygroClip(C) RelativeHumidity(%) SpecificHumidity(g/kg)'+ $
               ' WindSpeed(m/s) WindDirection(d) SensibleHeatFlux(W/m2) LatentHeatFlux(W/m2) ShortwaveRadiationDown(W/m2) ShortwaveRadiationDown_Cor(W/m2)' + $
               ' ShortwaveRadiationUp(W/m2) ShortwaveRadiationUp_Cor(W/m2) Albedo_theta<70d LongwaveRadiationDown(W/m2) LongwaveRadiationUp(W/m2) HeightSensorBoom(m)' + $
               ' LatitudeGPS_HDOP<1(degN) LongitudeGPS_HDOP<1(degW) ElevationGPS_HDOP<1(m) FanOK(%)'
        for i=0,n_elements(month_cent_m)-1 do printf,lun3,format=format_m,year_m[i],month_m[i],month_cent_m[i], $
           p_m[i],T_m[i],Thc_m[i],RH_cor_m[i],q_m[i],WS_m[i],WD_m[i],SHF_m[i],LHF_m[i],SRin_m[i],SRin_cor_m[i],SRout_m[i],SRout_cor_m[i],albedo_m[i], $
           LRin_m[i],LRout_m[i],Haws_m[i],GPSlat_m[i],GPSlon_m[i],GPSelev_m[i],fanOK_m[i]
        free_lun,lun3
     endelse
;----------End updating data files-------------------------------------------------------------------
; Plotting ------------------------------------------------------------------------------------------

     set_plot,'ps'

;!p.background = 255
;!p.color = 0
;!p.thick = 2
;!p.charsize = 1
;!p.charthick = 2
;!x.range = [0,24]
;!y.range = [0,200]
;!x.style = 1
;!y.style = 1
;!x.thick = 2
;!y.thick = 2
;!x.tickformat = '(f6.1)'
;!x.title=''
;!y.title=''
;!x.margin=[6,1]
;!y.margin=[3,1]
;!x.ticklen = 0.06
;!y.ticklen = 0.01
;!x.minor = 6
;!x.tickinterval = 6
;!y.minor = 2
;!y.tickinterval = 50
;!y.margin=[4,2]
     !y.style = 16              ; = ynozero

     A = FINDGEN(17) * (!PI*2/16.)
     USERSYM, COS(A), SIN(A)

     !p.multi = [0,1,5,0,0]

     device,filename=dir+datadir+AWS[i_AWS]+'_day'+version_no+'.ps',xsize=18,ysize=25,yoffset=2

     plot,day_cent_d/365.25+2000,p_d,min_value=-998,ytitle='Air pressure (hPa)'
     oplot,(month_cent_m-0.5)/12.+2000,p_m,psym=8

     plot,day_cent_d/365.25+2000,T_d,min_value=-998,ytitle='Air temperature PT100 (!eO!nC)'
     oplot,(month_cent_m-0.5)/12.+2000,T_m,psym=8
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,Thc_d,min_value=-998,ytitle='Air temperature HygroClip (!eO!nC)'
     oplot,(month_cent_m-0.5)/12.+2000,Thc_m,psym=8
     oplot,[2007,2025],[0,0],linestyle=1

;plot,day_cent_d/365.25+2000,RH_d,min_value=-998,ytitle='Relative humidity uncor (%)'

     plot,day_cent_d/365.25+2000,RH_cor_d,min_value=-998,ytitle='Relative humidity (%)'
     oplot,(month_cent_m-0.5)/12.+2000,RH_cor_m,psym=8

     plot,day_cent_d/365.25+2000,q_d,min_value=-998,ytitle='Specific humidity (g kg!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,q_m,psym=8

     plot,day_cent_d/365.25+2000,WS_d,min_value=-998,ytitle='Wind speed (m s!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,WS_m,psym=8

     plot,day_cent_d/365.25+2000,WD_d,min_value=-998,ytitle='Wind direction (!eO!n)',yrange=[0,360]
     oplot,(month_cent_m-0.5)/12.+2000,WD_m,psym=8

     plot,day_cent_d/365.25+2000,SHF_d,min_value=-998,ytitle='Sensible heat flux (W m!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,SHF_m,psym=8
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,LHF_d,min_value=-998,ytitle='Latent heat flux (W m!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,LHF_m,psym=8
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,SRin_d,min_value=-998,ytitle='Shortwave radiation down uncor (W m!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,SRin_m,psym=8
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,SRin_cor_d,min_value=-998,ytitle='Shortwave radiation down cor (W m!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,SRin_cor_m,psym=8

     plot,day_cent_d/365.25+2000,SRout_d,min_value=-998,ytitle='Shortwave radiation up uncor (W m!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,SRout_m,psym=8
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,SRout_cor_d,min_value=-998,ytitle='Shortwave radiation up cor (W m!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,SRout_cor_m,psym=8

     plot,day_cent_d/365.25+2000,albedo_d,min_value=-998,ytitle='Surface albedo'
     oplot,(month_cent_m-0.5)/12.+2000,albedo_m,psym=8

     plot,day_cent_d/365.25+2000,LRin_d,min_value=-998,ytitle='Longwave radiation down (W m!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,LRin_m,psym=8

     plot,day_cent_d/365.25+2000,LRout_d,min_value=-998,ytitle='Longwave radiation up (W m!e-1!n)'
     oplot,(month_cent_m-0.5)/12.+2000,LRout_m,psym=8

     plot,day_cent_d/365.25+2000,CloudCov_d,min_value=-998,ytitle='Cloud cover'

     plot,day_cent_d/365.25+2000,Tsurf_d,min_value=-998,ytitle='Surface temperature (!eO!nC)'
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,Haws_d,min_value=-998,ytitle='Height of sensor boom (m)'
;oplot,(month_cent_m-0.5)/12.+2000,Haws_m,psym=8

     plot,day_cent_d/365.25+2000,Hstk_d,min_value=-998,ytitle='Height of stake assembly (m)'
;oplot,(month_cent_m-0.5)/12.+2000,Hstk_m,psym=8

     plot,day_cent_d/365.25+2000,Hpt_d,min_value=-998,max_value=30,ytitle='Depth of pressure transducer cor&uncor (m)'
;oplot,(month_cent_m-0.5)/12.+2000,Hpt_m,psym=8
     oplot,day_cent_d/365.25+2000,Hpt_cor_d,min_value=-998
;oplot,(month_cent_m-0.5)/12.+2000,Hpt_cor_m,psym=8

     plot,day_cent_d/365.25+2000,ablation_pt_d,min_value=-998,max_value=200,ytitle='Ablation pressure transducer (mm)'

     plot,day_cent_d/365.25+2000,Tice1_d,min_value=-998,ytitle='Ice temperature (!eO!nC)'
     oplot,day_cent_d/365.25+2000,Tice2_d,min_value=-998
     oplot,day_cent_d/365.25+2000,Tice3_d,min_value=-998
     oplot,day_cent_d/365.25+2000,Tice4_d,min_value=-998
     oplot,day_cent_d/365.25+2000,Tice5_d,min_value=-998
     oplot,day_cent_d/365.25+2000,Tice6_d,min_value=-998
     oplot,day_cent_d/365.25+2000,Tice7_d,min_value=-998
     oplot,day_cent_d/365.25+2000,Tice8_d,min_value=-998
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,tiltX_d,min_value=-998,ytitle='Tilt of station (!eO!n)',yrange=[-30,30]
     oplot,day_cent_d/365.25+2000,tiltY_d,min_value=-998
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,GPSlat_d,min_value=-998,ytitle='GPS latitude (degN)',psym=8

     plot,day_cent_d/365.25+2000,GPSlon_d,min_value=-998,ytitle='GPS longitude (degW)',psym=8

     plot,day_cent_d/365.25+2000,GPSelev_d,min_value=-998,ytitle='GPS elevation (m)',psym=8

     plot,day_cent_d/365.25+2000,GPShdop_d,min_value=-998,ytitle='GPS HDOP',psym=8

     plot,day_cent_d/365.25+2000,Tlog_d,min_value=-998,ytitle='Logger temperature (!eO!nC)'
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,Ifan_d,min_value=-998,ytitle='Current drawn by fan (mA)'
     oplot,[2007,2025],[0,0],linestyle=1

     plot,day_cent_d/365.25+2000,Vbat_d,min_value=-998,ytitle='Battery voltage (V)'

     device,/close

;----------------------------------------------------------------------------------------------------

     beep
     print,'Done with this station. Run time (minutes) = ', (SYSTIME(1) - starttime)/60.
     print,'-----------------------------------------------------------'
  endfor
END
