web1: https://www.geodesy.linz.govt.nz/concord/
web2: http://www.zonums.com/online/coords/cotrans.php?module=13

Steps to convert NZGTM to LatLon: 

(1) Split in batch of 50.000 coords (just coords, no elev) 
(2) Transform NZGM to UTM in web1 (each batch)
(3) Transform UTM to lan lon in web2 (each batch)
(4) Merge batch of latlon 
(5) Add elev. 