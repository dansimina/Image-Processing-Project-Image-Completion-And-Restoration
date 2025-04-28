// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <random>

const int PATCH_SIZE = 13;
const int PATCH_RADIUS = PATCH_SIZE / 2;
const int DELTA = PATCH_RADIUS / 2;

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


int computeSSE(const Mat& img, const std::vector<std::vector<bool>>& mask, int x1, int y1, int x2, int y2) {
	int sum = 0;
	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
			if (!mask[y1 + dy][x1 + dx]) { 
				const Vec3b& p1 = img.at<Vec3b>(y1 + dy, x1 + dx);
				const Vec3b& p2 = img.at<Vec3b>(y2 + dy, x2 + dx);
				sum += (p1[0] - p2[0]) * (p1[0] - p2[0])
					+ (p1[1] - p2[1]) * (p1[1] - p2[1])
					+ (p1[2] - p2[2]) * (p1[2] - p2[2]);
			}
		}
	}
	return sum;
}

std::pair<int, int> propagate(const Mat& img, const std::vector<std::vector<bool>>& mask, int x, int y, int offsetX, int offsetY) {
	const int offsets[8][2] = { {-1,0}, {1,0}, {0,-1}, {0,1}, {-1,-1}, {1,1}, {-1,1}, {1,-1} };
	int bestSSE = computeSSE(img, mask, x, y, offsetX, offsetY);
	int bestX = offsetX, bestY = offsetY;

	for (int i = 0; i < 8; i++) {
		int nx = x + offsets[i][0];
		int ny = y + offsets[i][1];
		if (nx >= PATCH_RADIUS && nx < img.cols - PATCH_RADIUS && ny >= PATCH_RADIUS && ny < img.rows - PATCH_RADIUS && isValidPatch(mask, nx, ny)) {
			int sse = computeSSE(img, mask, x, y, nx, ny);
			if (sse < bestSSE) {
				bestSSE = sse;
				bestX = nx;
				bestY = ny;
			}
		}
	}
	return { bestX, bestY };
}

std::pair<int, int> findBestMatch(const Mat& img, const std::vector<std::vector<bool>>& mask, int x, int y) {
	int startX = STEP;
	int endX = img.cols - 1 - STEP;
	int startY = STEP;
	int endY = img.rows - 1 - STEP;

	int step = STEP;
	int bestMatch = INT_MAX;
	int bestX = startX, bestY = startY;

	std::vector<std::pair<int, int>> offsets = generateRandomPairs(mask, startX, endX, startY, endY, step);

	for (const auto& offset : offsets) {
		double diff = computeSSE(img, mask, x, y, offset.first, offset.second);
		if (diff < bestMatch) {
			bestMatch = diff;
			bestX = offset.first;
			bestY = offset.second;
		}
	}

	const int THRESHOLD = 0.1 * bestMatch;

	for (int i = 0; i < 3; i++) {
		int width = (endX - startX) / 4;
		int height = (endY - startY) / 4;

		step /= 2;

		startX = max(STEP, bestX - width);
		startY = max(STEP, bestY - height);
		endX = min(img.cols - 1 - STEP, bestX + width);
		endY = min(img.rows - 1 - STEP, bestY + height);

		offsets = generateRandomPairs(mask, startX, endX, startY, endY, step);

		for (const auto& offset : offsets) {
			double diff = computeSSE(img, mask, x, y, offset.first, offset.second);
			if (diff < bestMatch) {
				bestMatch = diff;
				bestX = offset.first;
				bestY = offset.second;
			}
		}
	}

	return propagate(img, mask, x, y, bestX, bestY);
}

void completePatch(Mat& img, std::vector<std::vector<bool>>& mask, int x, int y) {
	std::pair<int, int> offset = findBestMatch(img, mask, x, y);

	for (int dy = -DELTA; dy <= DELTA; dy++) {
		for (int dx = -DELTA; dx <= DELTA; dx++) {
			int targetY = y + dy;
			int targetX = x + dx;
			int sourceY = offset.second + dy;
			int sourceX = offset.first + dx;

			if (mask[targetY][targetX]) {
				img.at<Vec3b>(targetY, targetX) = img.at<Vec3b>(sourceY, sourceX);
				mask[targetY][targetX] = false;
			}
		}
	}

	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
			if (abs(dx) >= DELTA || abs(dy) >= DELTA) {
				int targetY = y + dy;
				int targetX = x + dx;
				int sourceY = offset.second + dy;
				int sourceX = offset.first + dx;

				if (!mask[targetY][targetX]) {
					// Calculate distance from patch center
					double dist = sqrt(dx * dx + dy * dy) / PATCH_RADIUS;

					// Calculate weight with smoother falloff
					double weight = pow(dist, 1.5); // Adjust exponent for transition sharpness
					weight = min(1.0, max(0.0, weight));

					// Get gradient information
					Vec3b sourcePixel = img.at<Vec3b>(sourceY, sourceX);
					Vec3b targetPixel = img.at<Vec3b>(targetY, targetX);

					// Calculate gradient-aware blending
					for (int c = 0; c < 3; c++) {
						// Blend with respect to local gradient
						img.at<Vec3b>(targetY, targetX)[c] =
							sourcePixel[c] * (1 - weight) + targetPixel[c] * weight;
					}
				}
			}
		}
	}
}

Mat imageReconstruction(Mat& img, int startX, int startY, int endX, int endY)
{
	Mat reconstruction = img.clone();

	startX = max(PATCH_RADIUS, startX);
	startY = max(PATCH_RADIUS, startY);
	endX = min(img.cols - 1 - PATCH_RADIUS, endX);
	endY = min(img.rows - 1 - PATCH_RADIUS, endY);

	std::vector<std::vector<bool>> mask = computeMask(img, startX, startY, endX, endY);

	for (int y = startY + 1; y <= endY - 1; y++) {
		for (int x = startX + 1; x <= endX - 1; x++) {
			if (mask[y][x]) {
				reconstruction.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
			}
		}
	}
	imshow("Region to Fill", reconstruction);
	waitKey(0);

	std::queue<std::pair<int, int>> Q;
	Q.push({ startX + 1, startY + 1 });
	Q.push({ endX - 1, startY + 1 });
	Q.push({ startX + 1, endY - 1 });
	Q.push({ endX - 1, endY - 1 });

	while (!Q.empty()) {
		auto front = Q.front();
		Q.pop();
		int x = front.first;
		int y = front.second;

		if (mask[y][x] && x >= startX && y >= startY && x <= endX && y <= endY) {
			completePatch(reconstruction, mask, x, y);

			if (x - DELTA - 1 >= 0 && mask[y][x - DELTA - 1]) {
				Q.push({ x - DELTA - 1, y });
			}
			if (y - DELTA - 1 >= 0 && mask[y - DELTA - 1][x]) {
				Q.push({ x, y - DELTA - 1 });
			}
			if (x + DELTA + 1 < reconstruction.cols && mask[y][x + DELTA + 1]) {
				Q.push({ x + DELTA + 1, y });
			}
			if (y + DELTA + 1 < reconstruction.rows && mask[y + DELTA + 1][x]) {
				Q.push({ x, y + DELTA + 1 });
			}
		}
	}

	for (int y = startY; y <= endY; y++) {
		for (int x = startX; x <= endX; x++) {
			if (mask[y][x]) {
				completePatch(reconstruction, mask, x, y);
			}
		}
	}

	return reconstruction;
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