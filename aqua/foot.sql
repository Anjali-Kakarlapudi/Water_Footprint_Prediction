-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Apr 06, 2024 at 03:48 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `foot`
--

-- --------------------------------------------------------

--
-- Table structure for table `adoption`
--

CREATE TABLE `adoption` (
  `id` int(11) NOT NULL DEFAULT 0,
  `name` varchar(100) NOT NULL,
  `image` varchar(100) NOT NULL,
  `area` varchar(100) NOT NULL,
  `tag` varchar(100) NOT NULL,
  `description` varchar(200) NOT NULL,
  `address` varchar(200) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `adoption`
--

INSERT INTO `adoption` (`id`, `name`, `image`, `area`, `tag`, `description`, `address`) VALUES
(127, 'Tanuja', '127.jpg', 'Vizag', '9848022222', '5 feet height, short hair, green dress, doll in hand', 'MVP Colony'),
(133, 'Siri', '133.jpg', 'Vijayawada', '8123456790', 'Height: 3\'2, talkative, mole on nose, ', 'Gopalapatnam'),
(138, 'Nani', '138.jpg', 'Guntur', '9999999999', 'Looking smart,active, red shoes, gold ring(letter N)', 'NAD');

-- --------------------------------------------------------

--
-- Table structure for table `feedback`
--

CREATE TABLE `feedback` (
  `userid` varchar(10) NOT NULL,
  `feedback` varchar(2000) NOT NULL,
  `date` varchar(50) NOT NULL,
  `time` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `feedback`
--

INSERT INTO `feedback` (`userid`, `feedback`, `date`, `time`) VALUES
('1234', 'hahaha', '2020-07-12', '13:19:17'),
('1001', 'Awesome application..it helps me a lot...Thank you team', '2020-07-17', '20:45:56'),
('1234', 'Awesome application..it helps me a lot...Thank you team', '2020-07-19', '21:32:59'),
('1234', 'Awesome application..it helps me a lot...Thank you team.................Amazing.This is my feedback..thanku', '2020-07-19', '21:44:52'),
('1234', 'ok its good', '2020-07-20', '10:03:44'),
('1234', 'thank you guys', '2020-07-20', '10:04:16');

-- --------------------------------------------------------

--
-- Table structure for table `person`
--

CREATE TABLE `person` (
  `id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `image` varchar(100) NOT NULL,
  `area` varchar(100) NOT NULL,
  `tag` varchar(100) NOT NULL,
  `description` varchar(200) NOT NULL,
  `address` varchar(200) NOT NULL,
  `status` varchar(30) NOT NULL DEFAULT 'Not Match'
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `person`
--

INSERT INTO `person` (`id`, `name`, `image`, `area`, `tag`, `description`, `address`, `status`) VALUES
(134, 'Anjali', '134.jpg', 'East Godavari', '8123456790', 'Dark eyes, 4 feet height, chain with letter R', 'Rajamundry', 'Match'),
(137, 'Unknown', '137.jpg', 'Vizag', '9876543110', 'found umbrella with her, mole on right hand', 'Maddilapalem', 'Not Match'),
(139, 'sashank', '139.jpg', 'West Godavari', '9876543110', 'Habit of keeping everything in mouth, talktive', 'Penugonda', 'Match');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `userid` int(11) NOT NULL,
  `email` varchar(50) NOT NULL,
  `password` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`userid`, `email`, `password`) VALUES
(7, 'rampraveenreddy4@gmail.com', '0007'),
(1234, '17131a05f1@gvpce.ac.in', '1234'),
(1924, '17131a05e8@gvpce.ac.in', '1924'),
(1999, '17131a05f0@gvpce.ac.in', '1999'),
(2000, '17131a05d5@gvpce.ac.in', '2000'),
(5678, 'padalarampraveenreddy.si20@iacademia.in', '5678'),
(9090, '17131a05f3@gvpce.ac.in', '9090'),
(9999, 'rampraveenreddy04@gmail.com', '9999'),
(17132, 'siva@gmail.com', '12345');

-- --------------------------------------------------------

--
-- Table structure for table `visitedplaces`
--

CREATE TABLE `visitedplaces` (
  `uid` int(11) NOT NULL,
  `age` int(11) NOT NULL,
  `area` varchar(200) NOT NULL,
  `places` varchar(2000) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `visitedplaces`
--

INSERT INTO `visitedplaces` (`uid`, `age`, `area`, `places`) VALUES
(1001, 23, 'visakhapatnam', 'hillstation,visakhapatnam, Simhachalam Rd, Simhachalam, Visakhapatnam'),
(1001, 23, 'visakhapatnam', 'hillstation,visakhapatnam,Bheemunipatnam'),
(1001, 23, 'visakhapatnam', 'hillstation,visakhapatnam,near Bheemunipatnam');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `person`
--
ALTER TABLE `person`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`userid`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `person`
--
ALTER TABLE `person`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=140;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `userid` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=17133;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
