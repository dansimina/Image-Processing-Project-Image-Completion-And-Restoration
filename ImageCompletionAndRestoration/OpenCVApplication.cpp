// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <random>

const int PATCH_SIZE = 7;
const int PATCH_RADIUS = PATCH_SIZE / 2;
const int DELTA = PATCH_RADIUS / 2;
const int GOOD_MATCH_THRESHOLD = 1000;

const int MAX_AREA = 360000;

const int STEP = 32;

wchar_t* projectPath;

struct SelectionData {
	Mat original;
	bool update = false;
	bool selected = false;
	int startX = 0;
	int startY = 0;
	int endX = 0;
	int endY = 0;
	int minX = 0;
	int minY = 0;
	int maxX = 0;
	int maxY = 0;
};

struct PatchPriority {
	int x, y;
	double priority;

	bool operator<(const PatchPriority& other) const {
		return priority < other.priority;
	}
};

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	SelectionData* data = (SelectionData*)param;
	Mat img = data->original.clone();

	if (event == EVENT_LBUTTONDOWN && !data->selected)
	{
		if (x >= data->minX && x <= data->maxX && y >= data->minY && y <= data->maxY) {
			data->update = true;
			data->startX = x;
			data->startY = y;
		}
	}
	else if (event == EVENT_MOUSEMOVE && data->update)
	{
		if (x >= data->minX && x <= data->maxX && y >= data->minY && y <= data->maxY) {
			data->endX = x;
			data->endY = y;

			img = data->original.clone();

			rectangle(img, Point(data->startX, data->startY),
				Point(data->endX, data->endY),
				Scalar(0, 255, 0), 2);

			imshow("My Window", img);
		}
	}
	else if (event == EVENT_LBUTTONUP && data->update)
	{
		if (x >= data->minX && x <= data->maxX && y >= data->minY && y <= data->maxY) {
			data->update = false;
			data->selected = true;
			data->endX = x;
			data->endY = y;

			img = data->original.clone();

			rectangle(img, Point(data->startX, data->startY),
				Point(data->endX, data->endY),
				Scalar(0, 255, 0), 2);

			imshow("My Window", img);
		}
	}
}

std::vector<std::vector<bool>> computeMask(Mat img, int startX, int startY, int endX, int endY) {
	std::vector<std::vector<bool>> mask(img.rows, std::vector<bool>(img.cols, false));

	startX = max(PATCH_RADIUS, startX);
	startY = max(PATCH_RADIUS, startY);
	endX = min(img.cols - 1 - PATCH_RADIUS, endX);
	endY = min(img.rows - 1 - PATCH_RADIUS, endY);

	for (int i = startY; i <= endY; i++) {
		for (int j = startX; j <= endX; j++) {
			mask[i][j] = true;
		}
	}

	return mask;
}

bool isValidPatch(const std::vector<std::vector<bool>>& mask, int x, int y) {
	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
			if (mask[y + dy][x + dx]) {
				return false;
			}
		}
	}

	return true;
}

std::vector<std::pair<int, int>> generateRandomPairs( const std::vector<std::vector<bool>>& mask, int searchStartX, int searchEndX, int searchStartY, int searchEndY, int step)
{
	if (mask.empty() || mask[0].empty() || searchStartX > searchEndX || searchStartY > searchEndY) {
		return {};
	}

	std::vector<std::pair<int, int>> result;
	const int RANGE = STEP / 2 - PATCH_RADIUS;

	srand(time(0));

	for (int y = searchStartY; y <= searchEndY; y += step) {
		for (int x = searchStartX; x <= searchEndX; x += step) {
			int dx = x + (rand() % (RANGE * 2)) - RANGE;
			int dy = y + (rand() % (RANGE * 2)) - RANGE;

			if (isValidPatch(mask, dx, dy)) {
				result.push_back({ dx, dy });
			}
		}
	}
	
	return result;
}


int computeSSE(const Mat& img, const std::vector<std::vector<bool>>& mask, int x1, int y1, int x2, int y2, int bestSoFar) {
	int sum = 0;
	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
			if (!mask[y1 + dy][x1 + dx]) { 
				const Vec3b& p1 = img.at<Vec3b>(y1 + dy, x1 + dx);
				const Vec3b& p2 = img.at<Vec3b>(y2 + dy, x2 + dx);
				sum += (p1[0] - p2[0]) * (p1[0] - p2[0])
					+ (p1[1] - p2[1]) * (p1[1] - p2[1])
					+ (p1[2] - p2[2]) * (p1[2] - p2[2]);

				if (sum > bestSoFar) {
					return sum;
				}
			}
		}
	}
	return sum;
}

std::pair<int, int> propagate(const Mat& img, const std::vector<std::vector<bool>>& mask, int x, int y, std::vector<std::vector<std::pair<int, int>>>& offsetMap) {
	int bestMatch = INT_MAX;
	int bestX = -1, bestY = -1;

	for (int dy = -DELTA; dy <= DELTA; dy++) {
		for (int dx = -DELTA; dx <= DELTA; dx++) {
			if (!(dx == 0 && dy == 0)) {
				int offsetX = offsetMap[y + dy][x + dx].first;
				int offsetY = offsetMap[y + dy][x + dx].second;

				if (offsetX != -1 && offsetY != -1 && isValidPatch(mask, offsetX, offsetY)) {
					int diff = computeSSE(img, mask, x, y, offsetX, offsetY, bestMatch);
					if (diff < bestMatch) {
						bestMatch = diff;
						bestX = offsetX;
						bestY = offsetY;
					}
				}
			}
		}
	}

	return { bestX, bestY };
}

std::pair<int, int> findBestMatch(const Mat& img, const std::vector<std::vector<bool>>& mask, int x, int y, std::vector<std::vector<std::pair<int, int>>>& offsetMap) {
	int startX = STEP;
	int endX = img.cols - 1 - STEP;
	int startY = STEP;
	int endY = img.rows - 1 - STEP;

	int step = STEP;
	int bestMatch = INT_MAX;
	int bestX = -1, bestY = -1;

	std::pair<int, int> result = propagate(img, mask, x, y, offsetMap);
	bestX = result.first;
	bestY = result.second;

	if (bestMatch < GOOD_MATCH_THRESHOLD && bestX != -1 && bestY != -1) {
		return { bestX, bestY };
	}

	std::vector<std::pair<int, int>> offsets = generateRandomPairs(mask, startX, endX, startY, endY, step);

	for (const auto& offset : offsets) {
		double diff = computeSSE(img, mask, x, y, offset.first, offset.second, bestMatch);
		if (diff < bestMatch) {
			bestMatch = diff;
			bestX = offset.first;
			bestY = offset.second;
		}
	}

	for (int i = 0; i < 5; i++) {
		int width = (endX - startX) / 4;
		int height = (endY - startY) / 4;

		step /= 2;

		startX = max(STEP, bestX - width);
		startY = max(STEP, bestY - height);
		endX = min(img.cols - 1 - STEP, bestX + width);
		endY = min(img.rows - 1 - STEP, bestY + height);

		offsets = generateRandomPairs(mask, startX, endX, startY, endY, step);

		for (const auto& offset : offsets) {
			double diff = computeSSE(img, mask, x, y, offset.first, offset.second, bestMatch);
			if (diff < bestMatch) {
				bestMatch = diff;
				bestX = offset.first;
				bestY = offset.second;
			}
		}
	}

	return { bestX, bestY };
}

bool isBoundaryPatch(const std::vector<std::vector<bool>>& mask, int x, int y) {
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			if (!(dx == 0 && dy == 0)) {
				int nx = x + dx;
				int ny = y + dy;

				if (!mask[ny][nx]) {
					return true;
				}
			}
		}
	}

	return false;
}

void completePatch(Mat& img, std::vector<std::vector<bool>>& mask, int x, int y, std::vector<std::vector<std::pair<int, int>>>& offsetMap) {
	std::pair<int, int> offset = findBestMatch(img, mask, x, y, offsetMap);

	offsetMap[y][x] = offset;

	for (int dy = -DELTA; dy <= DELTA; dy++) {
		for (int dx = -DELTA; dx <= DELTA; dx++) {
			int targetY = y + dy;
			int targetX = x + dx;
			int sourceY = offset.second + dy;
			int sourceX = offset.first + dx;

			if (mask[targetY][targetX]) {
				img.at<Vec3b>(targetY, targetX) = img.at<Vec3b>(sourceY, sourceX);
				mask[targetY][targetX] = false;
				offsetMap[y][x] = { targetX, targetY };
			}
		}
	}
}

double computePriority(Mat& img, const std::vector<std::vector<bool>>& mask, int x, int y) {
	double data = 0.0;
	int boundaryPixels = 0;
	double confidence = 0.0;
	int totalPixels = 0;
	int validPixels = 0;

	int averageR = 0, averageB = 0, averageG = 0;
	int meanAbsoluteError = 0;

	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
			totalPixels++;

			int ny = y + dy;
			int nx = x + dx;

			if (!mask[ny][nx]) {
				validPixels++;

				averageR += img.at<Vec3b>(ny, nx)[2];
				averageG += img.at<Vec3b>(ny, nx)[1];
				averageB += img.at<Vec3b>(ny, nx)[0];
			}
		}
	}

	confidence = (double) validPixels / totalPixels;

	averageR /= totalPixels;
	averageG /= totalPixels;
	averageB /= totalPixels;

	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
			int ny = y + dy;
			int nx = x + dx;

			if (!mask[ny][nx]) {
				meanAbsoluteError += abs(averageR - img.at<Vec3b>(ny, nx)[2]);
				meanAbsoluteError += abs(averageG - img.at<Vec3b>(ny, nx)[1]);
				meanAbsoluteError += abs(averageB - img.at<Vec3b>(ny, nx)[0]);
			}
		}
	}

	meanAbsoluteError /= (3 * 255 * totalPixels);

	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
			if (!mask[y + dy][x + dx]) {
				for (int ny = y + dy - 1; ny <= y + dy + 1; ny++) {
					for (int nx = x + dx - 1; nx <= x + dx + 1; nx++) {
						if (mask[ny][nx]) {
							data++;
							break;
						}
					}
				}
			}
		}
	}

	data /= totalPixels;

	return confidence * data + meanAbsoluteError;
}

Mat imageReconstruction(Mat& img, int startX, int startY, int endX, int endY)
{
	int n = 0;

	Mat reconstruction = img.clone();

	std::vector<std::vector<std::pair<int, int>>> offsetMap(img.rows,
		std::vector<std::pair<int, int>>(img.cols, { -1, -1 }));

	startX = max(PATCH_RADIUS + 1, startX);
	startY = max(PATCH_RADIUS + 1, startY);
	endX = min(img.cols - 2 - PATCH_RADIUS, endX);
	endY = min(img.rows - 2 - PATCH_RADIUS, endY);

	if (startX + DELTA >= endX || startY + DELTA >= endY) {
		return img;
	}

	std::vector<std::vector<bool>> mask = computeMask(img, startX, startY, endX, endY);

	for (int y = startY + 1; y <= endY - 1; y++) {
		for (int x = startX + 1; x <= endX - 1; x++) {
			if (mask[y][x]) {
				reconstruction.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
			}
		}
	}
	imshow("Region to Fill", reconstruction);
	waitKey(1);

	std::priority_queue<PatchPriority> Q;

	for (int y = startY; y <= endY; y++) {
		for (int x = startX; x <= endX; x++) {
			if (mask[y][x] && isBoundaryPatch(mask, x, y)) {
				double priority = computePriority(img, mask, x, y);
				Q.push({ x, y, priority });
			}
		}
	}

	while (!Q.empty()) {
		auto current = Q.top();
		Q.pop();
		int x = current.x;
		int y = current.y;

		if (mask[y][x] && x >= startX && y >= startY && x <= endX && y <= endY) {
			completePatch(reconstruction, mask, x, y, offsetMap);

			for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
				for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
					int nx = current.x + dx;
					int ny = current.y + dy;

					if (mask[ny][nx] && isBoundaryPatch(mask, nx, ny)) {
						double priority = computePriority(img, mask, nx, ny);
						Q.push({ nx, ny, priority });
					}
				}
			}
		}

		if (n == 20) {
			imshow("Region to Fill", reconstruction);
			waitKey(1);
			n = 0;
		}
		n++;
	}

	for (int y = startY; y <= endY; y++) {
		for (int x = startX; x <= endX; x++) {
			if (mask[y][x]) {
				completePatch(reconstruction, mask, x, y, offsetMap);
			}
		}
	}

	return reconstruction;
}

Mat performConvolutionOperation(Mat img, std::vector<std::vector<int>> kernel, int startX, int startY, int endX, int endY) {
	Mat result = img.clone();
	int kRows = kernel.size();
	int kCols = kernel[0].size();
	int kCenterY = kRows / 2;
	int kCenterX = kCols / 2;

	int S = 0;
	bool lowPass = true;

	for (int y = 0; y < kRows; y++) {
		for (int x = 0; x < kCols; x++) {
			S += kernel[y][x];
		}
	}

	for (int y = startY; y <= endY; y++) {
		for (int x = startX; x <= endX; x++) {
			int valR = 0, valG = 0, valB = 0;
			for (int m = 0; m < kRows; m++) {
				for (int n = 0; n < kCols; n++) {
					int ny = y + m - kCenterY;
					int nx = x + n - kCenterX;

					valB += kernel[m][n] * img.at<Vec3b>(ny, nx)[0];
					valG += kernel[m][n] * img.at<Vec3b>(ny, nx)[1];
					valR += kernel[m][n] * img.at<Vec3b>(ny, nx)[2];
				}
			}

			result.at<Vec3b>(y, x) = Vec3b(valB / S, valG / S, valR / S);
		}
	}

	return result;
}

Mat postprocessing(Mat img, int startX, int startY, int endX, int endY) {
	std::vector<std::vector<int>> gaussianFilter = {
		{1, 2, 1},
		{2, 4, 2},
		{1, 2, 1}
	};


	return performConvolutionOperation(img, gaussianFilter, startX, startY, endX, endY);
}

Mat preprocessing(Mat img) {
	int area = img.rows * img.cols;
	if (area < MAX_AREA) {
		return img;
	}

	double scale = sqrt((double)MAX_AREA / area);

	int newWidth = cvRound(img.cols * scale);
	int newHeight = cvRound(img.rows * scale);

	Mat resized;
	resize(img, resized, Size(newWidth, newHeight), 0, 0, INTER_AREA);

	return resized;
}

void testMouseClick()
{
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		// Set up the selection data
		SelectionData data;
		data.original = imread(fname);
		data.original = preprocessing(data.original);
		data.minX = PATCH_SIZE / 2;
		data.minY = PATCH_SIZE / 2;
		data.maxX = data.original.cols - PATCH_SIZE / 2 - 1;
		data.maxY = data.original.rows - PATCH_SIZE / 2 - 1;

		// Create a window
		namedWindow("My Window", 1);

		// Set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &data);

		// Show the original image
		imshow("My Window", data.original);

		// Wait until user press some key
		waitKey(0);

		if (data.selected) {
			int startX = min(data.startX, data.endX);
			int startY = min(data.startY, data.endY);
			int endX = max(data.startX, data.endX);
			int endY = max(data.startY, data.endY);

			Mat img = imageReconstruction(data.original, startX, startY, endX, endY);
			img = postprocessing(img, startX, startY, endX, endY);

			imshow("Selected Region", img);
			waitKey(0);
		}
	}
}


int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Demo\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testMouseClick();
			break;
		}
	} while (op != 0);
	return 0;
}