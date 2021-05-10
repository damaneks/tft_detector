# TFT Detector
Stop spending time and let your computer make VOD review

## Table of contents
* [General info](#general-info)
* [Demo](#demo)
* [Features](#features)

## General info
The application is used to detect characters from the game Teamfight Tactics using computer vision. For detection it uses trained Yolo v4 model. Collected data on the occurrence of given units in particular rounds and their positions can be later used for visualization and analysis. You can control your detection with simple Flask app hosted on your localhost. Just use a VPN and voilà, you can do it outside your home using only your smartphone.

## Demo
![demo-usage](img/demo.gif)

## Features
* Champions detection and preview from video
* Champions detection and preview from image

#### To-do
* Saving collected data with occurrence of units in particular rounds
* Saving collected data with position of units
* Game analysis summary
* Website with results of analysis best players' games
* Detection acceleration
* Items detecion
