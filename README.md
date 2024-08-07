# Masterarbeit

## Thesis title: 

### Integrated Parameter Study and Serviceability Assessment of Building Structures under Seismic Ground Motion using Stochastic Polynomial Chaos Expansion

Author: Wei-Teng, Kao 

Faculty: Chair of Structural Mechanics, Technical University of Munich

## Abstract
Geothermal energy is a renewable resource that currently draws significant attention in
Germany. However, its side effect, geothermal-induced earthquakes, must be considered
when planning a power plant near residential areas. In the thesis, two questions are answered.
First, will those earthquakes damage the building or make residents uncomfortable? Second,
what is the predominant seismic parameter causing the negative effect on the structures and
residents?

The following approaches are proposed to answer these two questions. A scenario-based
ground motion generator is introduced that models the seismic event in the Insheim area,
Germany. This generator can predict the representative ground motion with stochasticity
by seismic parameters such as moment magnitude and rupture distance. Once the generator
is built, the responses of the structure excited by the ground motion are simulated
and evaluated. Since the ground motions generated are stochastic and the material properties
of buildings are uncertain, a large amount of simulations has to be considered to
fully understand the relationship between ground motion and response, which will be highly
time-consuming. Therefore, a surrogate model, Stochastic Polynomial Chaos Expansion, is
utilized to efficiently interpret this relationship and capture the uncertain propagation in the
system.

Finally, the significance of each input seismic parameter will be analyzed using global sensitivity
analysis, and the potential damage and adverse effects on residents will be evaluated
using serviceability and comfortability analysis.

## Description: 

This is the source code for master thesis.

Please contact Wei-Teng, Kao via ge32gak@mytum.de for further information. 

In the repo, there are three main parts, which are:

1. [Scenario based ground motion generator (SBGMG)](./GroundMotion/GroundMotionGeneration/)
2. [Stochastic polynomial chaos expansion (SPCE)](./StochasticPCE/)
3. [Bulding's model written in ANSYS APDL](./ANSYS_simple_model/)

The application and examples are included in [this](./MA_application/) folder.

Please check further information in their corresponding README files. 









