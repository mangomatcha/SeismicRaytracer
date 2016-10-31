In seismology, the propagation of seismic waves from sources to receivers is 
studied to both locate the source of the waves and study the structure of the 
earth. The propagation of these waves is governed by the the same mechanical 
laws used in optics, so priciples from that field can be applied to the study 
of seismic waves (such as Snell's law and Huygens' principle). Seismic 
raytracing is an essential tool for earthquake location and involves solving 
the raypath equations between sources and receivers. This problem can be 
treated as a two-point initial value problem or boundary value problem and 
solved by finite-difference methods. The method I have implemented here is 
known as the "shooting" method, which fixes one end of the ray path and then 
uses the mechanical principles mentioned above and the incidence angle to find 
the coordinates of the endpoint. It traces the ray from a source, to a 
reflector at an arbitrary depth, to a receiver. This methods involves shooting 
a fan of rays from the fixed endpoint at various incidence angles which will 
bracket the location of the source.

The CPU implementation uses an iterative method to shoot narrower and narrower 
fans of rays until the calculated source location for one of the rays is 
adequately close to the true source location. The GPU implementation shoots a 
very dense fan of rays in parallel. It does not need to iterate through 
multiple fans of rays because the first fan is usually dense enough to return 
a result than is adequately close to the true source location. The program 
then uses Snell's law to calculate the travel time of the ray which came 
closest to the true source location.