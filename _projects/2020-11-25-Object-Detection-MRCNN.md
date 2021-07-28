---
title: "Object Detection using Mask RCNN"
excerpt: "The dual axis solar tracker is an Arduino project which shifts the position of a solar panel to the maximum intensity of sunlight thereby utilising optimum solar power . When implemented on a large scale, it is highly energy efficient."
toc: true
toc_sticky: true
toc_label: "Contents"

categories:
  - projects

tags:
  - Solar Tracker
  - BioMimicry
  - Efficient Technology
  - Clean Energy
  - Solar Energy
  - Arduino
  - LDR Sensor Technologies

last_modified_at: 2020-11-01T08:06:00-05:00
og_image: /assets/images/Arduino_1.jpg
header:
  teaser: "/assets/images/Arduino_1.jpg"
  overlay_image: /assets/images/Arduino_1.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Image credit: [**Harrison Broadbent**](https://unsplash.com/@harrisonbroadbent) on [**Unsplash**](https://unsplash.com/photos/19YCOjHosDk)"
  actions:
    - label: "View Code on Github"
      url: "https://github.com/khanfarhan10/DualAxisSolarTracker"

use_math: true
---

### Brief Summary of Project Working Prototype

Solar panels have been used increasingly in recent years to convert solar energy to electrical energy. The solar panel can be used either as a stand-alone system or as a large system that is connected to the electricity grid. The earth receives 84 Terawatts of power and our world consumes about 12 Terawatts of power every day. We are trying to extract more energy from the sun using solar panels. In order to maximize the conversion from solar to electrical energy, the solar panels have to be positioned perpendicular to the sun. Thus the tracking of the sun’s location and positioning of the solar panel are important. The goal of this project is to design an automatic tracking system, which can locate the position of the sun. The tracking system will align the solar panel so that it is positioned perpendicular to the sun for maximum energy conversion at all times. Photo-resistors will be used as sensors in this system. The system will consist of light sensing devices, microcontroller, gear motor system, and a solar panel. LDRs shall be used as the main light sensors. Two servo motors are to be fixed to the structure that holds the solar panel. The program for Arduino is to be uploaded to the microcontroller. LDRs sense the amount of sunlight falling on them. Four LDRs are divided into top, bottom, left and right. For east – west tracking, the analog values from two top LDRs and two bottom LDRs are compared and if the top set of LDRs receive more light, the vertical servo will move in that direction. If the bottom LDRs receive more light, the servo moves in that direction.

For angular deflection of the solar panel, the analog values from two left LDRs and two right LDRs are compared. If the left set of LDRs receive more light than the right set, the horizontal servo will move in that direction. If the right set of LDRs receive more light, the servo moves in that direction. Our system will provide an output with up to 40% more energy than the solar panels without tracking systems. The generation of power without using fossil fuels is the biggest challenge for the next half of this century. The idea of converting solar energy into electrical energy using photovoltaic panels holds its place in the front row as compared to other renewable sources. However the continuous change in the relative angle of the sun with reference to the earth reduces the watts delivered by the solar panel. In this context solar tracking system is the best alternative to increase the efficiency of the photovoltaic panel. Solar trackers move the payload towards the sun throughout the day.

### Future Prospects

As already mentioned earlier, a dual-axis solar tracker is much more efficient than a normal solar panel fixed at the same position all throughout the day since it can change its orientation depending upon the direction of the incident sunlight. The idea of a solar tracker can be well implemented in a country like India situated in the tropical region and thereby receiving most of the sunlight. Being able to rotate and change its position a solar tracker can provide maximum efficiency that can be achieved by a solar panel. We can overcome the problems of the thermal power through solar energy which is freely available at least for the next one million years. Therefore if we can properly utilize solar energy then the capability of generation centers might shoot up to such high levels that energy consumption in terms of electricity might get cheaper. However, maintenance is a crucial factor with respect to the installation of solar panels but still it is a way better alternative than burning our non-renewable sources of energy.

### PowerPoint Submission

You can view or <a href="https://github.com/khanfarhan10/DualAxisSolarTracker/blob/master/Dual%20Axis%20Solar%20Tracker%20-%20Submission%20Compressed.pptx" download>download this ppt</a>.

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTu7p0tt_GPxzyynqRnXLPsrb-VHfcxLq9lAJUudkU6dw7Q0T86wFE6TJwiOUEwGg/embed?start=true&loop=true&delayms=3000" frameborder="0" width="640" height="375" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<!--https://drive.google.com/drive/u/0/folders/1ARFyd7IyB1JBUPYUtg912D59EDsh4LqQ-->

### Complete Submitted Report

You can view or <a href="https://github.com/khanfarhan10/DualAxisSolarTracker/blob/master/Mechanical%20Project%20Solar%20Tracker%20Arduino%20.pdf" download>download this article</a>.

<iframe src="https://docs.google.com/viewer?srcid=1mTvfl8xbACls9sAvqIkXhu5YLJE8khSc&pid=explorer&efh=false&a=v&chrome=false&embedded=true" style="width:100%; height:900px;" frameborder="0" allowfullscreen></iframe>

<!--
https://drive.google.com/file/d/1JOVg2vlh7VMPx-X5GS4gC8UvLHzjIm6L/view?usp=sharing
https://drive.google.com/file/d/1mTvfl8xbACls9sAvqIkXhu5YLJE8khSc/view?usp=sharing
-->
