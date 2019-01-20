# boundgen
Algorithm to plan route over a course while simultaneously detecting cones.

## Method
In this algorithm detection and planning are combined in the same step.
It draws inspiration from a maximum-likelihood estimation of the most probable path given the currently observed cones and the previously observed cones.
At each step we generate a certain number of possible trajectories (all parts of a circle with different radii) and find that trajectory that has the highest value for the objective function.
This objective function is a measure of how close the generated path is to the midline of the track given the currently observed and previously observed cones.
The objective function is the sum¹ of the probability of the trajectory passing each cone at a distance of half the track width.

[1]: Note that we sum these probabilities, rather than multiply (which you would do in an MLE approach), because it was found that multiplication unfairly penalizes correct trajectories if other parts of the track are also detected.

## Usage
1. Generate a track using [trackgen](https://github.com/mopg/trackgen)
2. Plan trajectory over course from a certain starting point (make sure it is on the actual track).
3. Visualize result (either final trajectory or animation)

## Improve performance/accuracy
Several hyperparameters can be tuned to get better results. The algorithm seems to be pretty robust against changing these values.
1. `sigmabar`: standard deviation of cone location if next to car.
2. `alpha`: growth rate of standard deviation with distance from car. The larger this value is, the more important close cones become.
3. `sdetectmax`: maximum distance between car and cone to still influence path decision.
4. `Pcolcorr`: probability of estimating cone color correctly (note: for some reason it doesn't seem to work well if `Pcolcorr` is exactly `0.5`, but e.g., `0.51` would work)
5. `rmin`: minimum radius generated path can make.
6. `nsamples`: number of trial trajectories that are planned at each step.
7. `ds`: distance between detection points and trajectory updates.

## Example of planning and detection

<img src="img/track1.gif" alt="Example of planning and detection" width="225">
